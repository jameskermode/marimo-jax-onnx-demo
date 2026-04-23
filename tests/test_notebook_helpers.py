import pathlib

import numpy as np

from demo import build_grid, load_training_data


REPO = pathlib.Path(__file__).resolve().parent.parent


def test_load_training_data_local_file():
    path = REPO / "public" / "training_data.npz"
    x, y = load_training_data(str(path))
    assert x.shape == (200, 1)
    assert y.shape == (200, 1)
    assert x.dtype == np.float32
    assert y.dtype == np.float32


def test_load_training_data_file_url():
    path = REPO / "public" / "training_data.npz"
    url = path.as_uri()  # file:///...
    x, y = load_training_data(url)
    assert x.shape == (200, 1)
    assert y.dtype == np.float32


def test_build_grid_shape_and_dtype():
    g = build_grid(n_points=100, x_range=(-5.0, 5.0))
    assert g.shape == (100, 1)
    assert g.dtype == np.float32


def test_build_grid_endpoints_and_monotonic():
    g = build_grid(n_points=50, x_range=(-3.0, 4.0))
    assert float(g[0, 0]) == -3.0
    assert float(g[-1, 0]) == 4.0
    diffs = np.diff(g[:, 0])
    assert np.all(diffs > 0)
