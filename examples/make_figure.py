"""Render examples/gnome-compare.jpg, the README hero: the original, then textbook seam carving and dinocarve at
40% width, with a ring on the face so the difference reads at a glance.

  python examples/make_figure.py
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dinocarve.seams import carve, decode

HERE = Path(__file__).parent
FACE = (461, 422)                                   # a point on the gnome's face, in source pixels
RED, GREEN, INK, MUTED = (225, 45, 75), (22, 150, 80), (20, 24, 28), (110, 118, 128)
FONTS = ['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', '/Library/Fonts/Arial Bold.ttf', 'DejaVuSans-Bold.ttf']


def font(size):
    for f in FONTS:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


def load(d):
    return np.asarray(Image.open(d / 'image.jpg').convert('RGB')), decode(np.asarray(Image.open(d / 'rank.png').convert('RGB')))


def where(rank, k, x, y):
    """Position of source pixel (x, y) after removing k seams."""
    return int((rank[y, :x] >= k).sum()), y


def main(width=.4, height=900, gap=64, pad=56):
    img, ours = load(HERE / 'gnome')
    _, textbook = load(HERE / 'gnome-textbook')
    H, W = img.shape[:2]
    k = round(W * (1 - width))
    panels = [  # title, verdict, colour, pixels, face position after carving
        ('Original', f'{W}×{H}', MUTED, img, None),
        ('Classic', '✗ cuts the face', RED, carve(img, textbook, k), where(textbook, k, *FACE)),
        ('dinocarve', '✓ keeps the face', GREEN, carve(img, ours, k), where(ours, k, *FACE)),
    ]
    s = height / H
    tiles = [Image.fromarray(p).resize((round(p.shape[1] * s), height), Image.LANCZOS) for *_, p, _ in panels]
    top, label = pad + 70, 120
    sheet = Image.new('RGB', (sum(t.width for t in tiles) + gap * (len(tiles) - 1) + 2 * pad, top + height + label + pad), 'white')
    draw, x = ImageDraw.Draw(sheet), pad
    for (title, verdict, colour, _, face), t in zip(panels, tiles):
        sheet.paste(t, (x, top))
        if face:
            cx, cy, r = x + face[0] * s, top + face[1] * s, 92
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline='white', width=14)
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=colour, width=8)
        mid = x + t.width / 2
        draw.text((mid, top + height + 22), title, font=font(42), fill=INK, anchor='mt')
        draw.text((mid, top + height + 76), verdict, font=font(34), fill=colour, anchor='mt')
        x += t.width + gap
    # A bracket over the two carved panels: both have the same, much narrower width.
    x0, x1, y = pad + tiles[0].width + gap, x - gap, pad + 30
    draw.line((x0, y + 18, x0, y, x1, y, x1, y + 18), fill=MUTED, width=4)
    msg = f'Both resized to {round(width * 100)}% width'
    w = draw.textlength(msg, font=font(34))
    draw.rectangle(((x0 + x1 - w) / 2 - 16, y - 24, (x0 + x1 + w) / 2 + 16, y + 24), fill='white')
    draw.text(((x0 + x1) / 2, y), msg, font=font(34), fill=MUTED, anchor='mm')
    sheet.save(HERE / 'gnome-compare.jpg', quality=90)


if __name__ == '__main__':
    main()
