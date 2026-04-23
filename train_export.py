import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import onnxruntime as ort
import optax
from jax2onnx import to_onnx


class MLP(eqx.Module):
    layers: list

    def __init__(self, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.layers = [
            eqx.nn.Linear(1, 32, key=k1),
            eqx.nn.Linear(32, 32, key=k2),
            eqx.nn.Linear(32, 32, key=k3),
            eqx.nn.Linear(32, 1, key=k4),
        ]

    def __call__(self, x):
        # x: (batch, 1) -> (batch, 1). eqx.nn.Linear is per-sample, so vmap.
        def single(xi):
            h = xi
            for layer in self.layers[:-1]:
                h = jnp.tanh(layer(h))
            return self.layers[-1](h)
        return jax.vmap(single)(x)


def build_model(key):
    return MLP(key)


def _mse_loss(model, x, y):
    return jnp.mean((model(x) - y) ** 2)


def train(model, x, y, steps, lr):
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state, x, y):
        loss, grads = eqx.filter_value_and_grad(_mse_loss)(model, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    for _ in range(steps):
        model, opt_state, _ = step(model, opt_state, x, y)
    final_loss = float(_mse_loss(model, x, y))
    return model, final_loss


def export_onnx(model, path, opset=23):
    to_onnx(
        model,
        [("B", 1)],
        return_mode="file",
        output_path=str(path),
        opset=opset,
    )


def validate_parity(model, onnx_path, tol=1e-5):
    grid = np.linspace(-2 * math.pi, 2 * math.pi, 100, dtype=np.float32).reshape(-1, 1)
    jax_pred = np.asarray(model(jnp.asarray(grid)))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_pred = session.run(None, {input_name: grid})[0]
    max_diff = float(np.max(np.abs(jax_pred - onnx_pred)))
    assert max_diff < tol, f"JAX vs ONNX max diff {max_diff} exceeds tolerance {tol}"
    return max_diff


def make_data(seed, n=200):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2 * math.pi, 2 * math.pi, size=(n, 1)).astype(np.float32)
    noise = rng.normal(0.0, 0.1, size=(n, 1)).astype(np.float32)
    y = (np.sin(x) + noise).astype(np.float32)
    return x, y


def main():
    import pathlib

    public = pathlib.Path(__file__).parent / "public"
    public.mkdir(exist_ok=True)

    x_train, y_train = make_data(seed=42)
    model = build_model(jax.random.key(0))
    model, final_loss = train(
        model, jnp.asarray(x_train), jnp.asarray(y_train), steps=5000, lr=1e-3
    )
    print(f"final training loss: {final_loss:.6f} (noise floor ~0.01)")

    onnx_path = public / "model.onnx"
    export_onnx(model, str(onnx_path), opset=23)
    print(f"wrote {onnx_path} ({onnx_path.stat().st_size} bytes)")

    max_diff = validate_parity(model, str(onnx_path), tol=1e-5)
    print(f"JAX vs ONNX parity max|diff| = {max_diff:.2e}")

    npz_path = public / "training_data.npz"
    np.savez(npz_path, x_train=x_train, y_train=y_train)
    print(f"wrote {npz_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
