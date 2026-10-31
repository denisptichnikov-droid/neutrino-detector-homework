import numpy as np
from source import Source

src = Source(rng=42)


def test_shape_and_norm():
    d = src.directions(1.0)
    assert d.ndim == 2 and d.shape[1] == 3
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0)


def test_linear_yield():
    for E in (0.5, 1.0, 3.0):
        assert src.n_photons(E, fluctuate=False) == round(src.Y * E)


def test_poisson_mean():
    E, K = 1.0, 2000
    n = np.array([src.n_photons(E) for _ in range(K)])
    mean = src.Y * E
    assert abs(n.mean() - mean) < 5 * np.sqrt(mean / K)
    assert abs(n.var() / mean - 1) < 0.15


def test_isotropy():
    d = src.directions(100.0)            # ~1e6 фотонов
    M = len(d)
    assert np.all(np.abs(d.mean(axis=0)) < 5 / np.sqrt(3 * M))
    # <x^2> = <y^2> = <z^2> = 1/3 для изотропии
    assert np.allclose((d**2).mean(axis=0), 1 / 3, atol=5e-3)


def test_photons_generator():
    import types
    g = src.photons(0.1, fluctuate=False)
    assert isinstance(g, types.GeneratorType)
    d = np.array(list(g))
    assert d.shape == (1000, 3)
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0)


def test_photon_batches():
    batches = list(src.photon_batches(0.1234, fluctuate=False))   # 1234 фотона
    sizes = [len(b) for b in batches]
    assert sizes == [100] * 12 + [34]
    d = np.concatenate(batches)
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0)


def test_batches_total_is_poisson():
    E, K = 0.1, 3000
    n = np.array([sum(len(b) for b in src.photon_batches(E)) for _ in range(K)])
    mean = src.Y * E
    assert abs(n.mean() - mean) < 5 * np.sqrt(mean / K)
    assert abs(n.var() / mean - 1) < 0.1
