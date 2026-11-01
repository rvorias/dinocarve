# Living quilt (Seamframe export)

Static files only: upload this folder anywhere (it must be served over http(s), not file://).
Seams are precompiled; the page only decodes them (Web Worker) and carves on the GPU (WebGL2),
with a Canvas 2D fallback.

Fixed: Breathe mode, no ambient motion (hovering swells a tile), edge-matching order.

## Embed

```html
<iframe src="/seamframe-quilt/" title="Living quilt" loading="lazy"
        style="width:100%; height:70vh; border:0; border-radius:12px; display:block"></iframe>
```

Options (query string):
- `?c=` collection: cc0, kup (default cc0)
- `?bg=` page colour as hex without `#`, e.g. `?c=kup&bg=ffffff`

Collections: `cc0` (CC0 masterpieces), `kup` (Kup). CC0 artworks show title, artist and date
on hover (data/cc0/credits.json). Total size 47 MB; a visitor downloads only the tiles in view
first, then the rest in the background.

Rebuild: `python3 tools/export_quilt.py --collections cc0 kup` in the seamframe repo.
