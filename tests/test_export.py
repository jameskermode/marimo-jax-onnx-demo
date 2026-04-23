import jax
import onnx
import pytest

from train_export import build_model, export_onnx, validate_parity


@pytest.fixture
def tmp_model(tmp_path):
    model = build_model(jax.random.key(0))
    out_path = tmp_path / "m.onnx"
    export_onnx(model, str(out_path), opset=23)
    return out_path


def test_export_creates_file(tmp_model):
    assert tmp_model.exists()
    assert tmp_model.stat().st_size > 0


def test_exported_onnx_passes_checker(tmp_model):
    m = onnx.load(str(tmp_model))
    onnx.checker.check_model(m)


def test_exported_onnx_has_dynamic_batch(tmp_model):
    m = onnx.load(str(tmp_model))
    assert len(m.graph.input) == 1
    input_shape = m.graph.input[0].type.tensor_type.shape.dim
    # first dim should be symbolic (batch), second dim == 1
    assert input_shape[0].dim_param != "" or input_shape[0].dim_value == 0, (
        f"expected dynamic batch dim, got {input_shape[0]}"
    )
    assert input_shape[1].dim_value == 1


def test_parity_matches_jax(tmp_path):
    model = build_model(jax.random.key(0))
    path = tmp_path / "m.onnx"
    export_onnx(model, str(path), opset=23)
    max_diff = validate_parity(model, str(path), tol=1e-5)
    assert max_diff < 1e-5


def test_parity_raises_on_mismatch(tmp_path):
    # Export model A, then hand a *different* model (different seed) to validate_parity.
    model_a = build_model(jax.random.key(0))
    model_b = build_model(jax.random.key(999))
    path = tmp_path / "m.onnx"
    export_onnx(model_a, str(path), opset=23)
    with pytest.raises(AssertionError):
        validate_parity(model_b, str(path), tol=1e-5)
