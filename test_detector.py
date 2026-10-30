import numpy as np
from detector import Detector, fibonacci_sphere

det = Detector()


def random_on_sphere(M, R):
    return R * fibonacci_sphere(M)[np.random.permutation(M)]


def test_pmts_on_surface():
    assert np.allclose(np.linalg.norm(det.pmt_positions, axis=1), det.R)


def test_pmts_uniform():
    # центр масс ФЭУ около центра сферы, катоды не перекрываются
    assert np.linalg.norm(det.pmt_positions.mean(axis=0)) < 0.05 * det.R
    assert det.min_pmt_distance() > 2 * det.r


def test_pmt_center_belongs_to_itself():
    assert np.array_equal(det.pmt_index(det.pmt_positions), np.arange(det.N))


def test_edge_of_cathode():
    # точка на угловом расстоянии чуть меньше / больше края катода
    n = det.pmt_directions[0]
    t = np.cross(n, [0, 0, 1.]) if abs(n[2]) < 0.9 else np.cross(n, [1., 0, 0])
    t /= np.linalg.norm(t)
    alpha = np.arcsin(det.r / det.R)
    for da, expected in ((-1e-6, 0), (1e-6, -1)):
        p = det.R * (np.cos(alpha + da) * n + np.sin(alpha + da) * t)
        assert det.pmt_index(p)[0] == expected


def test_opposite_side_not_hit():
    assert det.pmt_index(-det.pmt_positions[0])[0] != 0


def test_coverage_monte_carlo():
    M = 2_000_000
    points = random_on_sphere(M, det.R)
    frac = det.is_on_pmt(points).mean()
    assert abs(frac - det.coverage()) < 5 * np.sqrt(frac / M)
