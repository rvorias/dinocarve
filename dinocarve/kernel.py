"""Fused Triton kernel for the minimum forward-energy vertical seam (DP forward pass + backtrack).

One program per batch item owns a whole row of columns and walks the rows sequentially, so a seam
costs one launch instead of ~H small PyTorch ops. Ties resolve in the order up, left, right,
matching seams.find_seams_torch up to floating-point ties."""
import torch
import triton
import triton.language as tl


@triton.jit
def _seam(cu, cl, cr, pen, back, m_scr, path, H, W, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < W
    base = b * H * W
    mrow = m_scr + b * W
    inf = float('inf')
    m = tl.load(cu + base + cols, mask=mask, other=inf) + tl.load(pen + base + cols, mask=mask, other=0.)
    m = tl.where(mask, m, inf)
    tl.store(mrow + cols, m, mask=mask)
    for y in range(1, H):
        tl.debug_barrier()
        left = tl.load(mrow + cols - 1, mask=mask & (cols > 0), other=inf)
        right = tl.load(mrow + cols + 1, mask=mask & (cols < W - 1), other=inf)
        off = base + y * W + cols
        vu = m + tl.load(cu + off, mask=mask, other=0.)
        vl = left + tl.load(cl + off, mask=mask, other=0.)
        vr = right + tl.load(cr + off, mask=mask, other=0.)
        best = vu
        arg = tl.zeros([BLOCK], dtype=tl.int32)
        arg = tl.where(vl < best, -1, arg)
        best = tl.minimum(best, vl)
        arg = tl.where(vr < best, 1, arg)
        best = tl.minimum(best, vr)
        m = tl.where(mask, best + tl.load(pen + off, mask=mask, other=0.), inf)
        tl.debug_barrier()
        tl.store(mrow + cols, m, mask=mask)
        tl.store(back + off, arg.to(tl.int8), mask=mask)
    tl.debug_barrier()
    x = tl.argmin(m, axis=0).to(tl.int32)
    tl.store(path + b * H + H - 1, x)
    for i in range(1, H):
        y = H - i
        x = x + tl.load(back + base + y * W + x).to(tl.int32)
        tl.store(path + b * H + y - 1, x)


def find_seams(cu, cl, cr, penalty):
    """[B,H,W] float32 CUDA tensors -> [B,H] int64 seam columns (on the GPU)."""
    cu, cl, cr, penalty = (t.float().contiguous() for t in (cu, cl, cr, penalty))
    B, H, W = cu.shape
    BLOCK = triton.next_power_of_2(W)
    back = torch.empty(B, H, W, dtype=torch.int8, device=cu.device)
    m = torch.empty(B, W, device=cu.device)
    path = torch.empty(B, H, dtype=torch.int32, device=cu.device)
    with torch.cuda.device(cu.device):  # Triton launches on the current device, not the tensors' device
        _seam[(B,)](cu, cl, cr, penalty, back, m, path, H, W, BLOCK=BLOCK, num_warps=min(16, max(4, BLOCK // 128)))
    return path.long()
