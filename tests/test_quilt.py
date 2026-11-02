import numpy as np
import torch

from dinocarve.quilt import compile_tile, edge_order, pack


def unpack(buf, K, R, offset):
    n = (R - 1 + 3) // 4
    size, paths = 2 + n, np.empty((K, R), np.int64)
    for k in range(K):
        b = buf[offset + k * size:offset + (k + 1) * size]
        c = np.frombuffer(b[2:], np.uint8)
        code = np.stack((c & 3, c >> 2 & 3, c >> 4 & 3, c >> 6 & 3), 1).reshape(-1)[:R - 1]
        paths[k] = int.from_bytes(b[:2], 'little') + np.concatenate(([0], np.cumsum(np.where(code == 1, 1, np.where(code == 2, -1, 0)))))
    return paths, offset + K * size


def removed_columns(paths, C):
    """Replay seams given in current coordinates; each must hit a live column in every row."""
    K, R = paths.shape
    rank = np.full((R, C), -1)
    for y in range(R):
        alive = list(range(C))
        for k in range(K):
            assert 0 <= paths[k, y] < len(alive)
            rank[y, alive.pop(paths[k, y])] = k
    return rank


def test_pack_round_trip():
    paths = np.array([[5, 6, 6, 5, 4, 4, 5], [0, 0, 1, 2, 2, 1, 0]])
    assert np.array_equal(unpack(pack(paths), 2, 7, 0)[0], paths)


def test_compile_tile_layout():
    rng = np.random.default_rng(0)
    rgba = np.concatenate((rng.integers(0, 256, (24, 40, 3), dtype=np.uint8), np.full((24, 40, 1), 255, np.uint8)), -1)
    blob, widths, Kx, Ky = compile_tile(rgba, 'textbook', levels=4, fraction=.5, device=torch.device('cpu'))
    assert (Kx, Ky) == (20, 12)
    assert widths == [40, 35, 30, 25, 20]
    v, off = unpack(blob, Kx, 24, 0)
    rank = removed_columns(v, 40)
    assert [(rank == k).sum() for k in range(Kx)] == [24] * Kx
    for wj in widths:
        h, off = unpack(blob, Ky, wj, off)          # horizontal seams of level j: rows = x, columns = y
        rank = removed_columns(h, 24)
        assert [(rank == k).sum() for k in range(Ky)] == [wj] * Ky
    assert off == len(blob)


def test_edge_order_follows_matching_edges():
    def tile(left, right):
        a = np.zeros((40, 16, 4), np.uint8)                       # 40 rows: not a multiple of the 27 profile bins
        a[:, :8, :3], a[:, 8:, :3], a[..., 3] = left, right, 255
        return a
    # right edge of 0 matches the left edge of 2, 2 -> 1, 1 -> 0
    order = edge_order([tile(10, 200), tile(100, 10), tile(200, 100)])
    assert order == [0, 2, 1]
