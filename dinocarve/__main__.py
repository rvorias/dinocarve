"""Compile an image into a seam order for <carved-image>.

  python -m dinocarve IMAGE OUT_DIR [--method dino|textbook] [--max-height 1024] [--fraction 0.6]

Writes OUT_DIR/{image.jpg, rank.png, carve.json} (+ saliency.png for --method dino)."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import seams

METHODS = {
    'dino': dict(forward='relative', stress=.5, keep_weight=10., budget=2., parts=12),
    'textbook': dict(forward='rubinstein'),
}


def compile_image(src, out, method='dino', max_height=1024, fraction=.6, device=None):
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    img = Image.open(src).convert('RGBA')
    if img.height > max_height:
        img = img.resize((round(img.width * max_height / img.height), max_height), Image.LANCZOS)
    rgba = np.array(img)
    H, W = rgba.shape[:2]
    K = int(W * fraction)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = METHODS[method]
    t = time.time()
    kw = dict(forward=cfg['forward'], stress=cfg.get('stress', 0.))
    if method == 'dino':
        from .dino import guidance
        sal, feats = guidance(rgba, device)
        Image.fromarray((sal.cpu().numpy() * 255).astype(np.uint8)).save(out / 'saliency.png')
        kw.update(keep=sal, keep_weight=cfg['keep_weight'], budget=cfg['budget'],
                  parts=seams.cluster(seams.pca(feats.float()), k=cfg['parts']))
    rank = seams.carve_order(seams.pixel_features(rgba), K, device=device, **kw)
    seconds = round(time.time() - t, 2)
    img.convert('RGB').save(out / 'image.jpg', quality=92)
    Image.fromarray(seams.encode(rank)).save(out / 'rank.png', optimize=True)
    meta = dict(version=1, width=W, height=H, max_remove=K, never=seams.NEVER, image='image.jpg', rank='rank.png',
                encoding='rank = R*256 + G. Pixel (x,y) is kept after removing k seams iff rank >= k.',
                method=method, params={k: v for k, v in cfg.items()}, seconds=seconds, device=str(device))
    (out / 'carve.json').write_text(json.dumps(meta, indent=1))
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('image')
    ap.add_argument('out')
    ap.add_argument('--method', choices=METHODS, default='dino')
    ap.add_argument('--max-height', type=int, default=1024)
    ap.add_argument('--fraction', type=float, default=.6, help='largest share of the width that can be removed')
    ap.add_argument('--device', default=None, help='cuda, cuda:1, cpu (default: cuda if available)')
    a = ap.parse_args()
    meta = compile_image(a.image, a.out, a.method, a.max_height, a.fraction, a.device)
    print(f"{a.out}: {meta['width']}x{meta['height']}, {meta['max_remove']} seams, {meta['method']}, {meta['seconds']}s on {meta['device']}")


if __name__ == '__main__':
    main()
