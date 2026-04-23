import math

import jax
import jax.numpy as jnp
import numpy as np

from train_export import build_model, make_data, train


def test_make_data_shape_dtype():
    x, y = make_data(seed=42)
    assert x.shape == (200, 1)
    assert y.shape == (200, 1)
    assert x.dtype == np.float32
    assert y.dtype == np.float32


def test_make_data_x_range():
    x, _ = make_data(seed=42)
    assert x.min() >= -2 * math.pi
    assert x.max() <= 2 * math.pi


def test_make_data_deterministic():
    x1, y1 = make_data(seed=42)
    x2, y2 = make_data(seed=42)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)


def test_make_data_custom_n():
    x, y = make_data(seed=7, n=50)
    assert x.shape == (50, 1)
    assert y.shape == (50, 1)


def test_build_model_forward_shape_dtype():
    model = build_model(jax.random.key(0))
    out = model(jnp.zeros((4, 1), dtype=jnp.float32))
    assert out.shape == (4, 1)
    assert out.dtype == jnp.float32


def test_build_model_param_count():
    import equinox as eqx

    model = build_model(jax.random.key(0))
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    count = sum(int(l.size) for l in leaves)
    # 3x32 tanh MLP, 1->1: 1*32+32 + 32*32+32 + 32*32+32 + 32*1+1 = 2209
    assert 2000 < count < 2500, f"param count was {count}"


def _mse(model, x, y):
    return float(jnp.mean((model(x) - y) ** 2))


def test_train_reduces_loss():
    x, y = make_data(seed=42)
    model = build_model(jax.random.key(0))
    x_j = jnp.asarray(x)
    y_j = jnp.asarray(y)
    before = _mse(model, x_j, y_j)

    trained, final_loss = train(model, x_j, y_j, steps=200, lr=1e-3)
    after = _mse(trained, x_j, y_j)

    assert after < 0.5 * before, f"loss did not drop enough: {before} -> {after}"
    # final_loss is the reported training MSE at the last step
    assert math.isfinite(final_loss)
    assert abs(final_loss - after) < 1e-4
