# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "plotly",
#     "anywidget",
#     "traitlets",
# ]
# ///

import io
import urllib.request

import numpy as np


def load_training_data(path_or_url):
    """Load x_train, y_train from an .npz file.

    Accepts a local filesystem path or any URL urllib can fetch (including
    file://). This indirection is what keeps the notebook working under
    Pyodide/WASM, where `mo.notebook_location()` returns a URL-like object
    rather than a `pathlib.Path`.
    """
    s = str(path_or_url)
    if "://" in s:
        with urllib.request.urlopen(s) as resp:
            data = resp.read()
    else:
        with open(s, "rb") as f:
            data = f.read()
    arrs = np.load(io.BytesIO(data))
    return arrs["x_train"], arrs["y_train"]


def build_grid(n_points, x_range):
    lo, hi = x_range
    return np.linspace(lo, hi, n_points, dtype=np.float32).reshape(-1, 1)


import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import io
    import time
    import urllib.request

    import anywidget
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import traitlets

    # The helpers below are duplicated from module top-level because marimo's
    # WASM runtime only executes cell bodies — module-level defs in demo.py
    # are invisible to Pyodide cells. Keeping both copies: tests import from
    # module top-level; cells receive these versions via reactive wiring.
    def load_training_data(path_or_url):
        s = str(path_or_url)
        if "://" in s:
            with urllib.request.urlopen(s) as resp:
                data = resp.read()
        else:
            with open(s, "rb") as f:
                data = f.read()
        arrs = np.load(io.BytesIO(data))
        return arrs["x_train"], arrs["y_train"]

    def build_grid(n_points, x_range):
        lo, hi = x_range
        return np.linspace(lo, hi, n_points, dtype=np.float32).reshape(-1, 1)

    return (
        anywidget,
        build_grid,
        go,
        load_training_data,
        mo,
        np,
        time,
        traitlets,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # JAX → ONNX → Marimo/WASM demo

        A ~2k-parameter MLP was trained in JAX to fit `y = sin(x)` and exported
        to ONNX. Inference runs **in your browser** via `onnxruntime-web`.
        Drag the sliders — predictions are computed client-side.
        """
    )
    return


@app.cell
def _(load_training_data, mo):
    training_url = str(mo.notebook_location() / "public" / "training_data.npz")
    x_train, y_train = load_training_data(training_url)
    return training_url, x_train, y_train


@app.cell
def _(anywidget, mo, traitlets):
    model_url = str(mo.notebook_location() / "public" / "model.onnx")

    class OnnxWidget(anywidget.AnyWidget):
        _esm = """
        const ORT_VERSION = "1.24.3";
        const ORT_URL = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.mjs`;
        const ORT_WASM_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

        const status = (el, msg, color) =>
            el.innerHTML = `<div style="font:13px/1.4 system-ui;color:${color||'#555'}">${msg}</div>`;

        async function render({ model, el }) {
            status(el, "Fetching onnxruntime-web runtime…");
            const ort = await import(ORT_URL);
            // Force single-threaded wasm: GitHub Pages does not serve
            // COOP/COEP, so SharedArrayBuffer is unavailable and the threaded
            // path would fall back anyway after an extra download.
            ort.env.wasm.numThreads = 1;
            ort.env.wasm.wasmPaths = ORT_WASM_BASE;

            const modelUrl = model.get("model_url");
            if (!modelUrl) { return status(el, "No model URL set.", "crimson"); }

            status(el, "Fetching model weights…");
            let session, backend;
            try {
                status(el, "Trying WebGPU execution provider…");
                session = await ort.InferenceSession.create(modelUrl, { executionProviders: ["webgpu"] });
                backend = "webgpu";
            } catch (e) {
                console.warn("webgpu EP unavailable, falling back to wasm:", e);
                try {
                    status(el, "WebGPU unavailable — loading single-threaded WASM backend…");
                    session = await ort.InferenceSession.create(modelUrl, { executionProviders: ["wasm"] });
                    backend = "wasm";
                } catch (err) {
                    return status(el, `ORT-Web failed to load model: ${err}`, "crimson");
                }
            }
            const inputName = session.inputNames[0];
            const outputName = session.outputNames[0];

            model.set("backend", backend);
            model.set("ready", true);
            model.save_changes();
            status(el, `onnxruntime-web ready (backend: <b>${backend}</b>)`, "#0a7a0a");

            model.on("change:input_x", async () => {
                const xs = model.get("input_x");
                if (!xs || xs.length === 0) {
                    model.set("output_y", []); model.save_changes(); return;
                }
                const arr = Float32Array.from(xs);
                const feeds = { [inputName]: new ort.Tensor("float32", arr, [xs.length, 1]) };
                const results = await session.run(feeds);
                model.set("output_y", Array.from(results[outputName].data));
                model.save_changes();
            });
        }
        export default { render };
        """

        model_url = traitlets.Unicode("").tag(sync=True)
        input_x = traitlets.List(traitlets.Float()).tag(sync=True)
        output_y = traitlets.List(traitlets.Float()).tag(sync=True)
        backend = traitlets.Unicode("").tag(sync=True)
        ready = traitlets.Bool(False).tag(sync=True)

    widget = mo.ui.anywidget(OnnxWidget(model_url=model_url))
    widget
    return OnnxWidget, model_url, widget


@app.cell
def _(mo):
    n_points = mo.ui.slider(50, 500, value=200, label="grid points")
    x_range = mo.ui.range_slider(-10, 10, value=(-8, 8), step=0.5, label="x range")
    mo.hstack([n_points, x_range])
    return n_points, x_range


@app.cell
def _(build_grid, n_points, np, time, widget, x_range):
    xs = build_grid(n_points.value, x_range.value)

    t0 = time.perf_counter()
    if widget.value.get("ready", False):
        widget.input_x = xs[:, 0].tolist()
        ys_flat = widget.value.get("output_y", [])
        # Drop the previous inference's output if it's still in flight — the
        # JS side hasn't yet processed our new input_x, so output_y is stale
        # and mis-shaped relative to xs. This cell will rerun when output_y
        # changes; until then the plot just skips the prediction line.
        if len(ys_flat) != xs.shape[0]:
            ys_flat = []
    else:
        ys_flat = []
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    ys = np.asarray(ys_flat, dtype=np.float32).reshape(-1, 1) if ys_flat else np.zeros((0, 1), dtype=np.float32)
    backend = widget.value.get("backend", "pending…")
    return backend, elapsed_ms, xs, ys


@app.cell
def _(backend, elapsed_ms, go, np, x_train, xs, y_train, ys):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_train[:, 0], y=y_train[:, 0], mode="markers",
        marker=dict(size=6, opacity=0.4), name="training data",
    ))
    true_xs = np.linspace(
        float(xs.min() if xs.size else -1),
        float(xs.max() if xs.size else 1),
        400,
    )
    fig.add_trace(go.Scatter(
        x=true_xs, y=np.sin(true_xs), mode="lines",
        line=dict(dash="dash", color="grey"), name="sin(x)",
    ))
    if ys.size:
        fig.add_trace(go.Scatter(
            x=xs[:, 0], y=ys[:, 0], mode="lines",
            line=dict(color="#ff7f0e", width=2), name="ONNX prediction",
        ))
    fig.update_layout(
        title=f"backend: {backend}   |   round-trip: {elapsed_ms:.1f} ms",
        xaxis_title="x", yaxis_title="y",
        height=420, margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(x=0.02, y=0.02),
        # Stable axes so the line smoothly updates instead of the plot bouncing.
        xaxis=dict(range=[-10, 10]),
        yaxis=dict(range=[-1.5, 1.5]),
    )
    fig
    return


if __name__ == "__main__":
    app.run()
