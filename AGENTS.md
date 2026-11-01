# Agent guide

Read `README.md` first for what the project does. This file covers how to change it safely.

## Commands

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python -m pytest tests -q                                  # CPU only, about 2 s, no downloads
.venv/bin/python -m dinocarve IMAGE OUT_DIR [--method dino|textbook] [--device cuda:1]
.venv/bin/python examples/make_figure.py                             # rebuilds examples/gnome-compare.jpg
python3 -m http.server 8000                                          # demo at /web/index.html
```

On a machine with several GPUs, pass `--device`. The Triton kernel launches on the tensors' device.

## Invariants

These hold for every change. Check them before you finish.

1. **The rank map contract.** For every k in [0, K], each row of `rank.png` has exactly one pixel with rank k for k < K, and exactly W−K pixels with rank 65535. `tests/test_seams.py::test_each_step_removes_one_pixel_per_row` checks this.
2. **Python and browser agree on the format.** `rank = R*256 + G`, RGB PNG, written by `seams.encode`, read by `web/dinocarve.js`. If you change one side, change the other side and the README table.
3. **Compiles are deterministic.** Two runs give byte-identical `rank.png` files. Do not use GPU float atomics (`index_add_`, `scatter_add_` on float sums with repeated indices) or randomized algorithms such as `torch.pca_lowrank` in the compile path. Use a seeded CPU generator for any randomness.
4. **WebGPU and CPU produce identical pixels.** The browser must never crop or stretch. Out of range, it letterboxes.
5. **The browser does no inference.** Everything model-related happens in the compiler.

## Where to change things

| Task | File |
|---|---|
| Cost weights (keep 10, budget 2, stress 0.5, 12 parts) | `METHODS` in `dinocarve/__main__.py` |
| A new cost term | `carve_order` in `dinocarve/seams.py`. Add it to `extra`, and update it after each removed seam the way `keep` and `parts` are. |
| A different DINO backbone | `BACKBONE` and `JAFAR_URL` in `dinocarve/dino.py`. JAFAR weights must match the backbone. The released checkpoints are listed in the JAFAR README. |
| Browser behaviour | `web/dinocarve.js`. Its header comment is the API reference. |

## Verification

- **Python changes.** Run the tests. Then compile `examples/gnome` again and compare it with the committed `rank.png`. A change in the result is expected only if you changed the method. Say so in the commit message.
- **Browser changes.** Open `web/index.html`. Check that the slider changes both panels and that resizing the window recarves the gallery. Check that `data-backend` on each `<carved-image>` shows the expected backend. For pixel checks, compare a rendered canvas with the reference rule "keep pixels with rank ≥ k" in JS.

## Gotchas

- The JAFAR checkpoints also pickle an omegaconf config. `dino.load_jafar_state` loads them with an allowlist unpickler. Do not switch to `weights_only=False`, and do not add omegaconf as a dependency.
- DINOv2 with registers has 4 register tokens after the class token. Patch tokens start at index `1 + num_register_tokens`.
- A canvas keeps its first context type (`webgpu` or `2d`). To switch backends, replace the canvas, as the fallback in `#render` does.
- Small float differences in the DINO features, for example from a different torch or hub version, can move individual seams. Determinism holds per environment, not across library versions.
