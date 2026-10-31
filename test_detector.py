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


def test_intersection_on_sphere():
    rng = np.random.default_rng(0)
    o = rng.uniform(-1, 1, (1000, 3))
    n = fibonacci_sphere(1000)
    p = det.intersect_surface(o, n)
    assert np.allclose(np.linalg.norm(p, axis=1), det.R)
    # точка лежит впереди по лучу
    assert np.all(np.sum((p - o) * n, axis=1) > 0)


def test_trace_towards_pmt():
    # из любой точки внутри луч точно на центр ФЭУ попадает в него
    o = np.array([0.3, -0.5, 0.7])
    n = det.pmt_positions - o
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    assert np.array_equal(det.trace(o, n), np.arange(det.N))


def test_trace_matches_disk_crossing():
    # прямая проверка: пересечение луча с плоскостью вписанного диска
    rng = np.random.default_rng(1)
    o = rng.uniform(-1, 1, 3)
    n = fibonacci_sphere(400_000)
    idx = det.trace(o, n)
    k = 0
    nk = det.pmt_directions[k]
    h = np.sqrt(det.R**2 - det.r**2)           # расстояние до плоскости диска
    t = (h - o @ nk) / (n @ nk)
    q = o + t[:, None] * n
    ok = (t > 0) & (np.linalg.norm(q - h * nk, axis=1) <= det.r)
    assert np.array_equal(ok, idx == k)


def test_photoelectrons_poisson():
    d = Detector(rng=3)
    pe = d.photoelectrons(np.full(200_000, 4))
    assert abs(pe.mean() - 4 * d.a) < 0.01
    assert abs(pe.var() - 4 * d.a) < 0.02
    assert np.all(d.photoelectrons(np.zeros(10, int)) == 0)


def test_survival_probability():
    d = Detector(attenuation_length=8.0, rng=7)
    for l in (0.5, 2.0, 4.0):
        frac = d.survives(np.full(500_000, l)).mean()
        assert abs(frac - np.exp(-l / 8.0)) < 0.003
    # без затухания выживают все
    assert det.survives(np.full(100, 3.0)).all()


def test_attenuation_from_center():
    # из центра путь до любого катода равен R
    d = Detector(attenuation_length=8.0, rng=8)
    n = fibonacci_sphere(2_000_000)
    hit_att = (d.trace(np.zeros(3), n) >= 0).mean()
    hit_free = (det.trace(np.zeros(3), n) >= 0).mean()
    assert abs(hit_att / hit_free - np.exp(-d.R / 8.0)) < 0.01
