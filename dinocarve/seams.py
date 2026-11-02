"""Nested seam carving: one compile produces a rank map that answers every narrower width.

rank[y, x] is the step at which pixel x of row y is removed (NEVER = kept at every supported
width). Removing k seams keeps exactly the pixels with rank >= k, so each width is a lookup
(Avidan & Shamir 2007, "multi-size images").

Cost of removing a pixel = relative forward energy (the edges the seam creates)
                         + stress     * seams already removed nearby (spreads seams out)
                         + keep_weight * saliency (DINO: what to keep)
                         + budget     * how over-carved this pixel's part already is (DINO parts)
"""
import numpy as np
import torch

NEVER = 65535


def pixel_features(rgba):
    """[H,W,4] uint8 -> [1,H,W,4] float premultiplied RGB + alpha in [0,1]."""
    x = torch.as_tensor(rgba, dtype=torch.float32)[None] / 255
    return torch.cat((x[..., :3] * x[..., 3:], x[..., 3:]), -1)


def forward_costs(x, mode='relative'):
    """Costs of a seam step (straight up, from up-left, from up-right) for [B,H,W,C] features.

    'rubinstein': each new adjacency costs its colour difference (Rubinstein et al. 2008).
    'relative': a new adjacency costs only as much as it differs from the nearest adjacency that
    already existed, so shifting a stroke is free and deleting a 1px stroke is not."""
    def d(a, b):
        return (a - b).norm(dim=-1)

    def shift(t, dim, k):
        n = t.shape[dim]
        if k < 0:
            return torch.cat((t.narrow(dim, 0, 1), t.narrow(dim, 0, n - 1)), dim)
        return torch.cat((t.narrow(dim, 1, n - 1), t.narrow(dim, n - 1, 1)), dim)

    left, right, up = shift(x, 2, -1), shift(x, 2, 1), shift(x, 1, -1)
    if mode == 'rubinstein':
        cu = d(right, left)
        return cu, cu + d(up, left), cu + d(up, right)
    ul, ur = shift(up, 2, -1), shift(up, 2, 1)
    dl, dr = d(x, left), d(x, right)
    dl[:, :, 0] = dr[:, :, 0]
    dr[:, :, -1] = dl[:, :, -1]
    cu = torch.minimum(dl, dr)
    return cu, cu + torch.minimum(d(up, ul), d(left, x)), cu + torch.minimum(d(up, ur), d(right, x))


def find_seams_torch(cu, cl, cr, penalty):
    """Batched minimum-cost vertical seam by dynamic programming: [B,H] column indices."""
    B, H, W = cu.shape
    inf = torch.full((B, 1), float('inf'), device=cu.device)
    m = cu[:, 0] + penalty[:, 0]
    back = torch.empty(B, H, W, dtype=torch.int8, device=cu.device)
    for y in range(1, H):
        st = torch.stack((torch.cat((inf, m[:, :-1]), 1) + cl[:, y], m + cu[:, y], torch.cat((m[:, 1:], inf), 1) + cr[:, y]))
        v, a = st.min(0)
        m = v + penalty[:, y]
        back[:, y] = a.to(torch.int8) - 1
    back = back.cpu().numpy()
    path = np.empty((B, H), np.int64)
    path[:, -1] = m.argmin(1).cpu().numpy()
    b = np.arange(B)
    for y in range(H - 1, 0, -1):
        path[:, y - 1] = path[:, y] + back[b, y, path[:, y]]
    return torch.tensor(path, device=cu.device)


def seam_solver(device):
    if device.type == 'cuda':
        try:
            from .kernel import find_seams
            return find_seams
        except ImportError:
            pass
    return find_seams_torch


def pca(f, q=16):
    """Project [...,C] features onto their top-q principal axes (exact, so compiles are reproducible)."""
    flat = f.reshape(-1, f.shape[-1])
    mean = flat.mean(0)
    x = flat - mean
    _, v = torch.linalg.eigh(x.T @ x / len(x))
    v = v[:, -q:].flip(1)
    v = v * torch.sign(v[v.abs().argmax(0), torch.arange(q)])
    return (f - mean) @ v


def cluster(features, k=12, iters=25, seed=0):
    """k-means part labels [H,W] from [H,W,C] features."""
    H, W, C = features.shape
    x = features.reshape(-1, C).float()
    g = torch.Generator(device='cpu').manual_seed(seed)
    c = x[torch.randperm(x.shape[0], generator=g)[:k].to(x.device)].clone()
    for _ in range(iters):
        onehot = torch.nn.functional.one_hot(torch.cdist(x, c).argmin(1), k).float()
        c = (onehot.T @ x) / onehot.sum(0).clamp_min(1)[:, None]  # matmul, not index_add_: GPU atomics vary the sum order
    return torch.cdist(x, c).argmin(1).view(H, W)


def carve_order(pixels, steps, forward='relative', stress=0., spread=8, keep=None, keep_weight=0., parts=None, budget=0.,
                device=None, seed=0, return_paths=False, row_mask=None):
    """Nested removal ranks for the first `steps` vertical seams: [H,W] uint16, or [B,H,W] for a batch.

    pixels: [B,H,W,C] features the join cost is measured in (see pixel_features). B > 1 carves B
        images of the same padded size in one pass, each with its own seams and statistics.
    keep: [H,W] or [B,H,W] importance in [0,1]; removing a pixel costs keep_weight * keep.
    parts: [H,W] or [B,H,W] integer labels; removing a pixel of part c costs
        budget * (fraction of c removed) / (fraction of the image removed), so no part is carved away whole.
    row_mask: [B,H] 0/1; masked rows cost nothing (padding when the batch has images of different heights).
    return_paths: also return the seams, [steps,H] (or [steps,B,H]), in the coordinates of the image at that step."""
    device = device or pixels.device
    x = pixels.to(device)
    B, H, L, _ = x.shape
    if steps > L - 2:
        raise ValueError('steps must leave at least 2 columns')
    batched = lambda t: t.to(device).expand(B, H, L) if t.dim() == 2 else t.to(device)
    seams = seam_solver(device)
    g = torch.Generator(device='cpu').manual_seed(seed)
    jitter = (torch.rand(B, H, L, generator=g) * 1e-6).to(device)
    orig = torch.arange(L, device=device).expand(B, H, L).contiguous()
    load = torch.zeros(B, H, L, device=device)
    rank = torch.full((B, H, L), NEVER, dtype=torch.int32, device=device)
    scale = forward_costs(x, forward)[0].mean((1, 2), keepdim=True).clamp_min(1e-8)
    keep = batched(keep).float() if keep is not None and keep_weight else None
    rm = row_mask.to(device).float()[:, :, None] if row_mask is not None else None
    if rm is not None:
        jitter = jitter * rm
    if parts is not None and budget:
        parts = batched(parts).long()
        P = int(parts.max()) + 1
        size = torch.zeros(B, P, device=device).scatter_add_(1, parts.reshape(B, -1), torch.ones(B, H * L, device=device)).clamp_min(1)
        removed = torch.zeros(B, P, device=device)
    else:
        parts = None
    paths = []
    for step in range(steps):
        cu, cl, cr = (c / scale for c in forward_costs(x, forward))
        W = orig.shape[2]
        extra = 0.
        if stress:
            extra = extra + stress * torch.nn.functional.avg_pool1d(load.reshape(B * H, 1, W), 2 * spread + 1, 1, spread, count_include_pad=False).reshape(B, H, W)
        if parts is not None:
            share = (removed / size) / max(step / L, 1e-3) / H
            extra = extra + budget * share.gather(1, parts.reshape(B, -1)).view(B, H, W)
        if keep is not None:
            extra = extra + keep_weight * keep
        cu, cl, cr = cu + extra, cl + extra, cr + extra
        if rm is not None:
            cu, cl, cr = cu * rm, cl * rm, cr * rm
        p = seams(cu, cl, cr, jitter[:, :, :W])[..., None]
        if return_paths:
            paths.append(p[..., 0].cpu())
        rank.scatter_(2, orig.gather(2, p), step)
        if stress:
            share_load = (load.gather(2, p) + 1) / 2
            load.scatter_add_(2, (p - 1).clamp(0, W - 1), share_load)
            load.scatter_add_(2, (p + 1).clamp(0, W - 1), share_load)
        if parts is not None:
            removed.scatter_add_(1, parts.gather(2, p)[..., 0], torch.ones(B, H, device=device))
        mask = torch.ones_like(orig, dtype=torch.bool)
        mask.scatter_(2, p, False)
        orig, load, x = orig[mask].view(B, H, W - 1), load[mask].view(B, H, W - 1), x[mask].view(B, H, W - 1, x.shape[-1])
        if parts is not None:
            parts = parts[mask].view(B, H, W - 1)
        if keep is not None:
            keep = keep[mask].view(B, H, W - 1)
    rank = rank.cpu().numpy().astype(np.uint16)
    paths = torch.stack(paths).numpy() if return_paths else None
    if B == 1:
        rank, paths = rank[0], (paths[:, 0] if return_paths else None)
    return (rank, paths) if return_paths else rank


def encode(rank):
    """[H,W] uint16 -> [H,W,3] uint8 lossless PNG payload: R = high byte, G = low byte."""
    img = np.zeros((*rank.shape, 3), np.uint8)
    img[..., 0] = rank >> 8
    img[..., 1] = rank & 255
    return img


def decode(img):
    return (img[..., 0].astype(np.uint16) << 8) | img[..., 1]


def carve(image, rank, k):
    """Apply a rank map: [H,W,...] image -> [H,W-k,...] with k seams removed."""
    keep = rank >= k
    return image[keep].reshape(image.shape[0], image.shape[1] - k, *image.shape[2:])
