"""Compile a folder of images into a living quilt for web/index.html.

  python -m dinocarve.quilt IMAGES OUT_DIR [--height 540] [--levels 32] [--fraction .55] [--device cuda:1]

Each image is scaled to --height and compiled for 2D carving, with both the dino method and the
textbook method (the quilt's Vanilla toggle):
- one nested vertical-seam order on the image (width),
- one nested horizontal-seam order for each of LEVELS+1 widths between W and the minimum width
  (height), compiled in one batch.
Tiles are ordered so each image's right edge continues into the next image's left edge
(optimal assignment of right edges to left edges, cycles concatenated).

Writes OUT_DIR/quilt.json and OUT_DIR/<tile>/{image.webp, seams.bin.gz, seams-textbook.bin.gz,
saliency.png}. A credits.json next to the images (a list of {file, title, artist, date}) is copied
and shown on hover."""
import argparse
import gzip
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import seams
from .__main__ import METHODS

FORMAT = ('seams.bin.gz (gzip): the vertical set (max_remove_x seams over height rows, width columns), then for each '
          'level j a horizontal set (max_remove_y seams over widths[j] rows, height columns). A seam is its uint16 LE '
          'first coordinate + 2-bit steps (0: 0, 1: +1, 2: -1), 4 per byte, LSB first, ceil((rows-1)/4) bytes.')


def pack(paths):
    """[K, R] seam coordinates -> bytes (see FORMAT)."""
    K, R = paths.shape
    n = (R - 1 + 3) // 4
    out = bytearray()
    for s in paths.astype(np.int64):
        d = np.diff(s)
        assert np.abs(d).max(initial=0) <= 1, 'seam is not connected'
        code = np.concatenate((np.where(d == 1, 1, np.where(d == -1, 2, 0)), np.zeros(n * 4 - len(d), np.int64))).astype(np.uint8).reshape(n, 4)
        out += int(s[0]).to_bytes(2, 'little') + bytes(code[:, 0] | code[:, 1] << 2 | code[:, 2] << 4 | code[:, 3] << 6)
    return bytes(out)


def assignment(cost):
    """Minimum-cost perfect matching (Hungarian algorithm): the column for each row."""
    n = len(cost)
    u, v, p, way = [0.] * (n + 1), [0.] * (n + 1), [0] * (n + 1), [0] * (n + 1)
    for i in range(1, n + 1):
        p[0], j0, minv, used = i, 0, [float('inf')] * (n + 1), [False] * (n + 1)
        while True:
            used[j0], i0, delta, j1 = True, p[j0], float('inf'), 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    out = [0] * n
    for j in range(1, n + 1):
        out[p[j] - 1] = j - 1
    return out


def edge_order(images, strip=8, bins=27):
    """Tile order in which right edges continue into left edges (colour profiles along the edge)."""
    profile = lambda a: np.stack([b.mean(0) for b in np.array_split(a[..., :3].astype(float).mean(1), bins)]) / 255
    left = [profile(a[:, :strip]) for a in images]
    right = [profile(a[:, -strip:]) for a in images]
    n = len(images)
    succ = assignment([[1e6 if i == j else np.abs(right[i] - left[j]).mean() for j in range(n)] for i in range(n)])
    seen, cycles = set(), []
    for s in range(n):
        c, x = [], s
        while x not in seen:
            seen.add(x)
            c.append(x)
            x = succ[x]
        if c:
            cycles.append(c)
    return [i for c in sorted(cycles, key=len, reverse=True) for i in c]


def compile_tile(rgba, method, levels, fraction, device, sal=None, parts=None):
    """Packed 2D seam orders for one image -> (bytes, widths, Kx, Ky)."""
    H, W = rgba.shape[:2]
    Kx, Ky = int(W * fraction), int(H * fraction)
    cfg = METHODS[method]
    kw = dict(forward=cfg['forward'], stress=cfg.get('stress', 0.), device=device)
    keep = sal if method == 'dino' else None
    parts = parts if method == 'dino' else None
    if method == 'dino':
        kw.update(keep_weight=cfg['keep_weight'], budget=cfg['budget'])
    rv, pv = seams.carve_order(seams.pixel_features(rgba), Kx, keep=keep, parts=parts, return_paths=True, **kw)
    widths = [W - round(j * Kx / levels) for j in range(levels + 1)]
    B = len(widths)
    px = seams.pixel_features(rgba)[0].to(device)                                   # [H,W,4]
    # Level j: the image carved to widths[j], transposed so horizontal seams run along rows; rows padded to W.
    P = torch.zeros(B, W, H, 4, device=device)
    Kp = torch.zeros(B, W, H, device=device)
    Pp = torch.full((B, W, H), int(parts.max()) + 1 if parts is not None else 0, dtype=torch.long, device=device)
    mask = torch.zeros(B, W, device=device)
    alive = torch.as_tensor(rv.astype(np.int64), device=device)
    for j, wj in enumerate(widths):
        k = alive >= W - wj
        P[j, :wj] = px[k].view(H, wj, 4).transpose(0, 1)
        if keep is not None:
            Kp[j, :wj] = keep.to(device)[k].view(H, wj).T
        if parts is not None:
            Pp[j, :wj] = parts.to(device)[k].view(H, wj).T
        mask[j, :wj] = 1
    _, ph = seams.carve_order(P, Ky, keep=Kp if keep is not None else None, parts=Pp if parts is not None else None,
                              row_mask=mask, return_paths=True, **kw)                 # [Ky, B, W]
    blob = pack(pv) + b''.join(pack(ph[:, j, :wj]) for j, wj in enumerate(widths))
    return blob, widths, Kx, Ky


def load(path, height, max_aspect=2.2):
    im = Image.open(path).convert('RGB')
    im = im.resize((round(im.width * height / im.height), height), Image.LANCZOS)
    if im.width > max_aspect * height:
        x = (im.width - int(max_aspect * height)) // 2
        im = im.crop((x, 0, x + int(max_aspect * height), height))
    return np.array(im.convert('RGBA'))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('images')
    ap.add_argument('out')
    ap.add_argument('--height', type=int, default=540)
    ap.add_argument('--levels', type=int, default=32)
    ap.add_argument('--fraction', type=float, default=.55, help='largest share of width and of height that can be removed')
    ap.add_argument('--limit', type=int, default=0, help='compile only the first N images')
    ap.add_argument('--device', default=None, help='cuda, cuda:1, cpu (default: cuda if available)')
    a = ap.parse_args()
    device = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    from .dino import guidance
    src, out = Path(a.images), Path(a.out)
    paths = sorted(p for p in src.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp'))[:a.limit or None]
    out.mkdir(parents=True, exist_ok=True)
    tiles, images, names = [], [], set()
    for path in paths:
        t0 = time.time()
        name = base = ''.join(c if c.isalnum() else '-' for c in path.stem.lower()).strip('-')[:40]
        while name in names:                                                       # truncated names can collide
            name = f'{base}-{len(names)}'
        names.add(name)
        rgba = load(path, a.height)
        H, W = rgba.shape[:2]
        dest = out / name
        dest.mkdir(exist_ok=True)
        Image.fromarray(rgba[..., :3]).save(dest / 'image.webp', quality=92, method=6)
        sal, feats = guidance(rgba, device)
        Image.fromarray((sal.cpu().numpy() * 255).round().astype(np.uint8)).save(dest / 'saliency.png', optimize=True)
        parts = seams.cluster(seams.pca(feats.float()), k=METHODS['dino']['parts'])
        for method, file in (('dino', 'seams.bin.gz'), ('textbook', 'seams-textbook.bin.gz')):
            blob, widths, Kx, Ky = compile_tile(rgba, method, a.levels, a.fraction, device, sal, parts)
            (dest / file).write_bytes(gzip.compress(blob, 9))
        tiles.append(dict(name=name, source=path.name, width=W, height=H, max_remove_x=Kx, max_remove_y=Ky, widths=widths,
                          energies=['smart', 'textbook']))
        images.append(rgba)
        print(f'{name}: {W}x{H}, {Kx} + {len(widths)}x{Ky} seams, {time.time() - t0:.1f}s', flush=True)
    order = edge_order(images)
    q = dict(version=2, height=a.height, levels=a.levels, energies=['smart', 'textbook'], tiles=tiles, order=order,
             orders=dict(edges=order), format=FORMAT)
    (out / 'quilt.json').write_text(json.dumps(q, separators=(',', ':')))
    if (src / 'credits.json').exists():
        shutil.copy2(src / 'credits.json', out / 'credits.json')
    print(f'{out}: {len(tiles)} tiles')


if __name__ == '__main__':
    main()
