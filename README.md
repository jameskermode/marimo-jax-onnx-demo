# marimo-jax-onnx-demo

A deliberately minimal end-to-end demo: a JAX-trained MLP is exported to ONNX
and runs **in the browser** via `onnxruntime-web` inside a Marimo WASM notebook.
No Python numerical runtime in the browser for model inference — Python just
orchestrates and plots.

The model learns `y = sin(x)` (~2.2k parameters, three tanh hidden layers).
The point is toolchain validation, not ML.

## Live demo

Deployed to GitHub Pages from `main` by `.github/workflows/deploy.yml` —
paste the Pages URL here after the first successful deploy.

## Reproduce locally

Requires [`uv`](https://docs.astral.sh/uv/).

1. **Regenerate the model** (rarely needed — `public/model.onnx` is committed):
   ```
   uv run --group train python train_export.py
   ```
   Trains in JAX, exports to ONNX, validates JAX vs `onnxruntime` parity
   (< 1e-5) before writing `public/model.onnx` and `public/training_data.npz`.

2. **Edit the notebook interactively:**
   ```
   uv run marimo edit demo.py
   ```

3. **Build the static WASM bundle** (what CI deploys):
   ```
   uv run marimo export html-wasm demo.py -o dist/ --mode run
   ```
   Serve `dist/` with any static HTTP server. The page prefers WebGPU, falls
   back to single-threaded WASM; the selected backend is shown in the plot
   title.

4. **Run the Python-side tests:**
   ```
   uv run pytest
   ```

## How it splits

| Side | What runs | Deps |
| ---- | --------- | ---- |
| `train_export.py` | JAX training + `jax2onnx` export + parity gate | `[train]` group in `pyproject.toml` |
| `demo.py` | Marimo notebook orchestrating an `anywidget` that calls `onnxruntime-web` from a CDN | base deps only; no JAX, no `onnxruntime` on the Python side |

CI never installs the `[train]` group — builds stay fast, and `model.onnx` is a
committed artifact rather than a build output.

## Design rationale

See [`SPEC.md`](SPEC.md) for the full spec, scope boundaries, and known
foot-guns (ONNX opset, SharedArrayBuffer headers, WebGPU operator support).

## A note on the `marimo<0.23` pin

`pyproject.toml` pins `marimo<0.23`. marimo 0.23 added a frontend check that
refuses to load anywidget `_esm` modules delivered as `data:` URLs. Under
Pyodide/WASM, marimo forces `virtual_file_storage=None`, which makes every
anywidget `_esm` come through as a `data:` URL — so every custom anywidget
in a WASM-exported notebook breaks. This repo needs custom anywidgets to
run `onnxruntime-web`, so we stay on 0.22 until marimo ships a frontend
bridge for `@file/` URLs under WASM or an opt-in trust flag. See the
comment block in `pyproject.toml`.

**Security implications of the pin.** marimo <=0.20.4 has a critical
pre-auth RCE ([CVE-2026-39987 / GHSA-2679-6mx9-h9xc](https://github.com/marimo-team/marimo/security/advisories/GHSA-2679-6mx9-h9xc)),
fixed in 0.23.0. The vulnerable endpoint (`/terminal/ws`) only exists in
`marimo edit`'s server. The exposure profile for this repo:

- **Deployed Pages artifact:** not affected — it's a static HTML bundle
  with no marimo server at runtime.
- **CI (`.github/workflows/deploy.yml`):** not affected — runs `marimo
  export html-wasm` (CLI), never `marimo edit`, in an ephemeral Actions
  sandbox.
- **Local `marimo edit demo.py`:** would be affected if bound to a public
  interface. Marimo's default bind is 127.0.0.1, which keeps the endpoint
  loopback-only. Do not pass `--host 0.0.0.0` on untrusted networks.

We're at 0.22.5, past CVE-2026-39987's vulnerable range (≤0.20.4). The
proxy-abuse advisory GHSA-xjv7-6w92-42r7 is fixed in 0.16.4 (well below us
too).
