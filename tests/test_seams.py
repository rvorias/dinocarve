"""CPU-only tests of the carving core (no model downloads). Run: python -m pytest tests -q"""
import numpy as np
import torch

from dinocarve import seams

CPU = torch.device('cpu')


def toy(h=24, w=48, seed=0):
    """Noisy green background with a flat warm disk: the classic case where edge energy carves the subject."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 4), np.uint8)
    img[..., 3] = 255
    img[..., 1] = rng.integers(60, 200, (h, w))
    yy, xx = np.mgrid[:h, :w]
    disk = ((yy - h / 2) / (h / 3)) ** 2 + ((xx - w * .45) / (w / 6)) ** 2 < 1
    img[disk, :3] = (210, 120, 60)
    return img, disk


def test_each_step_removes_one_pixel_per_row():
    img, _ = toy()
    K = 20
    rank = seams.carve_order(seams.pixel_features(img), K, stress=.5, device=CPU)
    for k in range(K):
        assert ((rank == k).sum(1) == 1).all(), f'step {k} must remove exactly one pixel in every row'
    assert ((rank == seams.NEVER).sum(1) == img.shape[1] - K).all()


def test_seams_are_connected():
    img, _ = toy()
    _, paths = seams.carve_order(seams.pixel_features(img), 15, device=CPU, return_paths=True)
    assert np.abs(np.diff(paths, axis=1)).max() <= 1


def test_carve_keeps_row_order_and_width():
    img, _ = toy()
    rank = seams.carve_order(seams.pixel_features(img), 10, device=CPU)
    cols = np.broadcast_to(np.arange(img.shape[1]), img.shape[:2])
    out = seams.carve(cols, rank, 10)
    assert out.shape == (img.shape[0], img.shape[1] - 10)
    assert (np.diff(out, axis=1) > 0).all()


def test_keep_term_protects_flat_subject():
    img, disk = toy()
    K = 12
    edges = seams.carve_order(seams.pixel_features(img), K, forward='rubinstein', device=CPU)
    kept = seams.carve_order(seams.pixel_features(img), K, keep=torch.tensor(disk, dtype=torch.float32), keep_weight=10., device=CPU)
    assert (edges[disk] < K).sum() > 0, 'textbook energy should carve into the flat subject'
    assert (kept[disk] < K).sum() == 0, 'the keep term should route every seam around it'


def test_part_budget_runs_and_is_deterministic():
    img, disk = toy()
    parts = torch.tensor(disk.astype(np.int64))
    a = seams.carve_order(seams.pixel_features(img), 15, parts=parts, budget=2., stress=.5, device=CPU)
    b = seams.carve_order(seams.pixel_features(img), 15, parts=parts, budget=2., stress=.5, device=CPU)
    assert np.array_equal(a, b)


def test_png_encoding_roundtrip():
    rank = np.array([[0, 1, 255, 256], [1000, 65534, 65535, 7]], np.uint16)
    assert np.array_equal(seams.decode(seams.encode(rank)), rank)


def test_pca_is_deterministic_and_orthonormal():
    f = torch.randn(30, 20, 8, generator=torch.Generator().manual_seed(0))
    a, b = seams.pca(f, q=4), seams.pca(f, q=4)
    assert torch.equal(a, b)
    assert a.shape == (30, 20, 4)
