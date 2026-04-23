# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

Greenfield. Only `SPEC.md` exists — it is the source of truth for what to build. Read it before making non-trivial changes; the sections below are a shortcut, not a replacement.

## What this project is

An end-to-end demo that a JAX-trained model can be exported to ONNX and executed in a Marimo WASM notebook via `onnxruntime-web`. The whole point is **toolchain validation**, not ML. The model is a deliberately tiny `sin(x)` MLP (~2.2k params).

## Two-artifact split (critical architectural constraint)

The repo is two programs that never run together:

1. **`train_export.py`** — offline, author-only. JAX + Equinox training, then `jax2onnx.to_onnx` emits `public/model.onnx`. Validates the exported ONNX against JAX outputs (`max|Δ| < 1e-5`) and exits non-zero on mismatch so broken exports never get committed. Also writes `public/training_data.npz`.
2. **`demo.py`** — Marimo notebook. **Must not import `jax`, `jaxlib`, or `onnxruntime`** on the Python side. Inference happens in the browser via `onnxruntime-web` loaded from a CDN inside an anywidget. Python only orchestrates and plots.

`model.onnx` is a **committed repo artifact**, not a build output. Regenerating it is a manual local step. CI never runs `train_export.py`.

This split is why there are two dependency groups in `pyproject.toml` — `[train]` (jax, equinox, optax, jax2onnx, onnxruntime, numpy) and the demo deps (marimo, numpy, matplotlib, anywidget, traitlets). The CI workflow installs the demo group **only**, keeping builds ~30s and JAX-free.

## The anywidget is the only non-trivial piece

Everything else is plumbing. The widget:
- dynamically imports `onnxruntime-web` from a **pinned** jsdelivr URL (floating versions drift and break silently),
- prefers the `webgpu` execution provider, falls back to `wasm`, reports which via a `backend` traitlet,
- uses a one-shot request/response pattern: Python sets `input_x` traitlet → JS `on("change:input_x")` runs the session → JS writes `output_y` back via `save_changes()`,
- gets its model URL as a traitlet set to `str(mo.notebook_location() / "public" / "model.onnx")`,
- should stay under ~100 lines of JS in `_esm`. If it grows, something's wrong.

## Path resolution quirk

`mo.notebook_location()` returns a `pathlib.Path` locally but a URL-like object under WASM. `str()` works for both in `fetch()` / widget URLs, but **do not call `.read_bytes()` on it** — route binary reads through `urllib.request` so Pyodide handles them.

`public/` is copied verbatim into the WASM bundle by `marimo export html-wasm`, which is why model + training data live there.

## Commands

Regenerate the model (local only, rarely needed):
```
uv run --group train python train_export.py
```

Edit the notebook:
```
uv run marimo edit demo.py
```

Build the static WASM bundle (what CI does):
```
uv run marimo export html-wasm demo.py -o dist/ --mode run
touch dist/.nojekyll
```

## Known foot-guns (from SPEC.md §"Likely failure modes")

- **Opset mismatch.** `jax2onnx` 0.9+ defaults to opset 23; `onnxruntime-web` only supports opset 23 from **1.23** onward. We pin `1.24.3` in `demo.py`. If the ORT-Web CDN version drifts below 1.23, model load fails with a cryptic numeric WASM error (e.g. `9000256`) — not an opset-named error.
- **SharedArrayBuffer / WASM threads.** Require COOP/COEP headers, which GitHub Pages doesn't set. Stay on single-threaded WASM as the fallback; WebGPU doesn't need those headers.
- **Activation choice.** `tanh` works on WebGPU. `gelu` is fragile — check the WebGPU supported-operators list before swapping.
