"""Render examples/gnome-compare.jpg: original | textbook at 50% | dinocarve at 50% | DINO saliency.

  python examples/make_figure.py
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from dinocarve.seams import carve, decode

HERE = Path(__file__).parent


def load(d):
    return np.asarray(Image.open(d / 'image.jpg').convert('RGB')), decode(np.asarray(Image.open(d / 'rank.png').convert('RGB')))


def main(width=.5, height=420, gap=12):
    img, ours = load(HERE / 'gnome')
    _, textbook = load(HERE / 'gnome-textbook')
    k = round(img.shape[1] * (1 - width))
    saliency = np.asarray(Image.open(HERE / 'gnome' / 'saliency.png').convert('RGB'))
    panels = [('original', img), ('textbook, 50% width', carve(img, textbook, k)),
              ('dinocarve, 50% width', carve(img, ours, k)), ('DINO saliency', saliency)]
    tiles = [Image.fromarray(p).resize((round(p.shape[1] * height / p.shape[0]), height), Image.LANCZOS) for _, p in panels]
    sheet = Image.new('RGB', (sum(t.width for t in tiles) + gap * (len(tiles) + 1), height + 2 * gap + 22), 'white')
    draw, x = ImageDraw.Draw(sheet), gap
    for (label, _), t in zip(panels, tiles):
        sheet.paste(t, (x, gap))
        draw.text((x, height + gap + 6), label, fill=(60, 60, 60))
        x += t.width + gap
    sheet.save(HERE / 'gnome-compare.jpg', quality=88)


if __name__ == '__main__':
    main()
