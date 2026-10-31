import numpy as np
from detector import Detector
from simulation import expected_fraction


def test_expected_fraction_matches_tracing():
    det = Detector(rng=0)
    rng = np.random.default_rng(0)
    M = 2_000_000
    phi = rng.uniform(0, 2 * np.pi, M)
    ct = rng.uniform(-1, 1, M)
    st = np.sqrt(1 - ct**2)
    n = np.stack([st * np.cos(phi), st * np.sin(phi), ct], axis=1)
    for origin in ((0, 0, 0), (0, 0, 1.0)):
        idx = det.trace(np.array(origin, float), n)
        mc = (idx >= 0).mean()
        assert abs(mc - expected_fraction(det, origin).sum()) < 5 * np.sqrt(mc / M)
