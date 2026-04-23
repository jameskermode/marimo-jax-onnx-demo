# Spec: Minimal Marimo + jax2onnx + onnxruntime-web demo

**Goal:** Demonstrate end-to-end that a JAX-trained model can be exported to ONNX and executed in a Marimo WASM notebook via `onnxruntime-web`, with no Python numerical runtime in the browser for the model forward pass.

**Scope:** Deliberately minimal. A single `y = sin(x)` regression problem, a tiny MLP, one interactive widget, one plot. The point is to de-risk the toolchain, not to showcase ML.

## Deliverables

Two artifacts in a single repo `marimo-jax-onnx-demo/`:

1. **`train_export.py`** — run once, offline. Trains the MLP in JAX and emits `model.onnx`.
2. **`demo.py`** — the Marimo notebook. Loads `model.onnx` in the browser via `onnxruntime-web`, runs inference reactively, plots predictions vs. training data.

A third artifact — `model.onnx` — is checked in (it's small, ~a few KB) so the Marimo notebook has no offline-build step from the reader's perspective.

## Repo layout

```
marimo-jax-onnx-demo/
├── pyproject.toml           # uv-managed, two dependency groups
├── README.md                # how to reproduce
├── train_export.py          # JAX training + ONNX export (offline)
├── demo.py                  # Marimo notebook (PEP 723 inline deps)
├── public/
│   ├── model.onnx           # exported network (weights baked in)
│   └── training_data.npz    # x_train, y_train for plotting overlay
└── .github/workflows/
    └── deploy.yml           # export html-wasm → GitHub Pages
```

Rationale for `public/`: Marimo's WASM exporter copies `public/` verbatim into the bundle, and `mo.notebook_location() / "public" / "model.onnx"` resolves correctly both locally and under `html-wasm`. No separate asset-hosting step.

## Model

A genuinely tiny MLP — the smaller the better for this demo, because the goal is toolchain validation, not ML:

- Input: `x`, shape `(batch, 1)`, float32
- Three hidden layers, 32 units each, `tanh` activation
- Output: `y`, shape `(batch, 1)`, float32
- Total parameters: ~2.2k
- Framework: **Equinox**. Reason: `jax2onnx` has first-class Equinox support with parity-tested exports, it's pure-JAX (no Flax state machinery), and it's the simplest thing that will export cleanly.

## `train_export.py` — specification

**Training setup:**
- Sample 200 points: `x_train ~ Uniform(-2π, 2π)`, `y_train = sin(x_train) + ε`, `ε ~ N(0, 0.1²)`.
- Fixed PRNG seed (42) for reproducibility.
- Optimiser: `optax.adam(1e-3)`.
- 5000 steps, full-batch, MSE loss.
- Print final loss; should be dominated by the noise floor (~0.01).

**Export step:**
```python
from jax2onnx import to_onnx

# model is an Equinox module
to_onnx(
    model,
    [("B", 1)],            # dynamic batch dimension
    return_mode="file",
    output_path="public/model.onnx",
    opset=23,              # jax2onnx 0.9 default
)
```

**Validation before exit:** Load the freshly-written ONNX with `onnxruntime` (native) and check `max(|jax_pred - onnx_pred|) < 1e-5` on a fresh grid of 100 points. Script exits non-zero if this fails — so a broken export never gets committed.

**Also save** `public/training_data.npz` containing `x_train`, `y_train` as float32 arrays, for overlay plotting.

**Dependencies** (one `pyproject.toml` group `[train]`, pinned):
```
jax >= 0.8.1
equinox
optax
jax2onnx >= 0.9.0
onnxruntime      # for native validation only
numpy
```

## `demo.py` — the Marimo notebook

**PEP 723 header:**
```python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo",
#   "numpy",
#   "matplotlib",
#   "anywidget",
#   "traitlets",
# ]
# ///
```

No `jax`, no `jaxlib`, no `onnxruntime` on the Python side. The browser does the inference; Python just orchestrates and plots. That's the whole point of the demo.

**Cell structure (roughly 6 cells):**

1. **Imports and constants.** `marimo as mo`, `numpy as np`, `matplotlib.pyplot as plt`, paths resolved with `mo.notebook_location()`.

2. **Load training data.** Read `training_data.npz` via `mo.notebook_location() / "public" / "training_data.npz"` — a URL under WASM, a Path locally. Use `numpy.load` on the bytes.

3. **The anywidget** — this is the only bespoke piece.

   - `_esm`: ES module that on first render dynamically imports `onnxruntime-web` from a CDN (jsdelivr: `https://cdn.jsdelivr.net/npm/onnxruntime-web@1.x/dist/ort.mjs`), creates an `InferenceSession` from `model.onnx` (URL passed in via traitlet), and exposes a `run(xs) → ys` path.
   - Synchronisation pattern: Python sets `input_x: List[float]` (traitlet); JS `model.on("change:input_x", ...)` runs the session and writes back `output_y: List[float]` via `model.save_changes()`. One-shot request/response per interaction.
   - Execution provider: prefer `webgpu`, fall back to `wasm`. Log which backend was selected into a `backend: str` traitlet so the notebook can display it.
   - Model URL is a traitlet set from Python to `str(mo.notebook_location() / "public" / "model.onnx")`.
   - Size budget: <100 lines of JS in `_esm`. If it gets bigger, something's wrong.

4. **Sliders for the plotting grid.**
   - `n_points = mo.ui.slider(50, 500, value=200, label="grid points")`
   - `x_range = mo.ui.range_slider(-10, 10, value=(-8, 8), step=0.5, label="x range")`

5. **Reactive inference cell.** Builds `xs` from the sliders, pushes to the widget, reads back `ys`. Handles the first-render race (widget not initialised yet) by checking `widget.value.get("ready", False)` and returning an empty array early so Marimo's reactivity settles cleanly.

6. **Plot cell.** Matplotlib scatter of `(x_train, y_train)`, line plot of ONNX predictions, dashed line of true `sin(x)` for reference, annotation showing which backend was selected (`webgpu` / `wasm`) and inference time (wall-clock round-trip from Python → JS → Python, measured in the inference cell).

## GitHub Actions deployment

Standard Marimo pattern (matches the Thomas Bury blog post referenced in prior research):

```yaml
# .github/workflows/deploy.yml
# Trigger: push to main
# Steps:
#   - checkout
#   - setup-uv
#   - uv sync (demo group only, NOT train group)
#   - uv run marimo export html-wasm demo.py -o dist/ --mode run
#   - touch dist/.nojekyll
#   - upload-pages-artifact, deploy-pages
```

The `train` group is explicitly excluded from CI so the workflow stays fast (~30s) and doesn't pull JAX. `model.onnx` is a repo artifact, not a build output — regenerating it is a manual, local step run by the author.

## Acceptance criteria

A reviewer can tell the demo works if, in this order:

1. `uv run --group train python train_export.py` produces `public/model.onnx` and exits 0 (validation passed).
2. `uv run marimo edit demo.py` opens the notebook locally; sliders update the prediction curve smoothly; the curve follows the scatter, flattening at the boundaries outside the training range.
3. `uv run marimo export html-wasm demo.py -o /tmp/out --mode run` produces a self-contained HTML file that, opened directly in Chrome, behaves identically to (2). The backend annotation reads `webgpu` on a machine with WebGPU enabled.
4. Opened in Firefox (where WebGPU is behind a flag by default), the same HTML file works with backend reading `wasm` — confirming the fallback path.
5. Served from GitHub Pages by the workflow, all of the above still holds.

## Things deliberately out of scope

- **WebGPU IO binding.** For a 2.2k-parameter MLP, the CPU↔GPU copy overhead dominates the actual compute. Not worth the complexity.
- **Model quantisation.** Pointless at this size.
- **Training in the browser.** That would need JAX in Pyodide, which is the whole problem we're avoiding.
- **Dynamic model loading.** One model, baked into `public/`. If we later want a "retrain with different noise level and re-export" workflow, that's a follow-up.
- **Error handling beyond basics.** If ORT-Web fails to load, the widget surfaces the error message in a red div and the Python side handles the empty-output case gracefully. That's it — no retries, no complicated UI.

## Likely failure modes to watch for

- **CDN version drift.** Pin the `onnxruntime-web` version in the `_esm` import URL; floating versions will eventually break the demo silently.
- **CORS on the CDN for SharedArrayBuffer.** If you want WASM threads, GitHub Pages needs `Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy` headers — which it doesn't set. Single-threaded WASM works without these; WebGPU doesn't need them. Stick to single-threaded WASM as the fallback to avoid this entire rabbit hole.
- **Opset mismatch.** `jax2onnx` defaults to opset 23 in 0.9+; older `onnxruntime-web` builds may not support it. Either pin ORT-Web to a recent version (≥1.20) or pass `opset=20` to `to_onnx` and let the release notes guide you if a primitive isn't covered.
- **`tanh` is fine; `gelu` is fragile.** If you later swap activations, stick to ops from the WebGPU supported-operators list (link in prior research) to keep the WebGPU path working.
- **`mo.notebook_location()` returning different types.** It's a `Path` locally and a `urllib` URL under WASM; `str()` on both gives something `fetch()` can consume, but avoid `.read_bytes()` calls on it — route binary reads through `urllib.request` which Pyodide handles.

## Estimated effort

With Claude Code: roughly 2–3 hours end-to-end including the GitHub Actions workflow and a README. The anywidget is the only piece that needs real thought; everything else is plumbing.