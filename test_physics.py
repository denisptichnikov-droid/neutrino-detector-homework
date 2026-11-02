"""Независимая проверка физики модели.

Ожидаемые значения считаются здесь заново: пересечение луча с плоским диском,
численный интеграл телесного угла по диску, прямая свёртка распределений.
Формулы из кода модели в расчёт ожиданий не входят. Допуски заданы заранее:
4 sigma для статистики, seed фиксированы. Что эти тесты действительно ловят
ошибки, проверяет check_mutations.py.
"""
import glob
import pathlib
import re

import nbformat
import numpy as np
import pymc as pm
import pytest
from scipy import stats
from scipy.special import logit, logsumexp

from detector import Detector, fibonacci_sphere
from pymc_model import multinomial_model, photon_model, sample_Q
from reconstruction import (expected_photons, neyman_logp, neyman_random,
                            reconstruct, reconstruction_model)
from simulation import expected_fraction, simulate_charge
from source import Source

R, N, r, a, Y, L = 2.0, 40, 0.10, 0.5, 10_000, 8.0
SIGMA = 4.0
P_MIN = 6e-5    # двусторонняя вероятность выхода за 4 sigma
HERE = pathlib.Path(__file__).parent


def isotropic(g, n):
    v = g.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def assert_mean(sample, expected):
    err = np.std(sample) / np.sqrt(len(sample))
    assert abs(np.mean(sample) - expected) < SIGMA * err, (np.mean(sample), err, expected)


def assert_variance(sample, expected):
    assert abs(np.var(sample) - expected) < SIGMA * expected * np.sqrt(2 / len(sample)), (np.var(sample), expected)


def assert_counts(observed, expected):
    chi2 = np.sum((observed - expected)**2 / expected)
    assert stats.chi2.sf(chi2, len(expected)) > P_MIN, chi2


def disc_hits(directions, o, u, R=R, r=r):
    """Номер диска, который пересекает луч, или -1: прямое пересечение с плоскостью диска."""
    D = np.sqrt(R**2 - r**2)
    un = u @ directions.T
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (D - directions @ o) / un
        rho2 = o @ o + 2 * t * (u @ o)[:, None] + t**2 - D**2    # на плоскости |p - D n|^2 = |p|^2 - D^2
        hit = (t > 0) & (rho2 <= r**2)
    return np.where(hit.any(axis=1), hit.argmax(axis=1), -1)


def disc_fraction(directions, o, L=np.inf):
    """Доля фотонов из o на каждом диске: интеграл cos(psi) exp(-d/L) / (4 pi d^2) по площади диска."""
    D = np.sqrt(R**2 - r**2)
    x, w = np.polynomial.legendre.leggauss(8)
    s, w = 0.5 * (x + 1), 0.5 * w                     # s = (rho/r)^2 равномерно по площади
    phi = (np.arange(16) + 0.5) * 2 * np.pi / 16
    out = np.zeros(len(directions))
    for k, n in enumerate(directions):
        e1 = np.cross(n, [0.3, 0.5, 0.81])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        rho = r * np.sqrt(s)[:, None, None]
        p = D * n + rho * (np.cos(phi)[None, :, None] * e1 + np.sin(phi)[None, :, None] * e2)
        v = p - o
        d = np.linalg.norm(v, axis=2)
        f = (v @ n) / (4 * np.pi * d**3) * np.exp(-d / L)
        out[k] = np.pi * r**2 * np.sum(w[:, None] / 16 * f)
    return out


def neyman_direct(m, lam):
    """log P(pe = m): прямая свёртка Poisson(h; lam) Poisson(m; a h)."""
    h = np.arange(0, int(lam + 20 * np.sqrt(lam) + 2 * m / a + 60))
    with np.errstate(divide='ignore'):
        return logsumexp(stats.poisson.logpmf(h, lam) + stats.poisson.logpmf(m, a * h))


# --- геометрия ---

@pytest.mark.parametrize('R_, N_', [(2.0, 40), (15.0, 2000)])
def test_pmts_are_uniform_on_the_sphere(R_, N_):
    det = Detector(R=R_, N=N_, r=r)
    assert np.allclose(np.linalg.norm(det.pmt_positions, axis=1), R_)
    d = np.linalg.norm(det.pmt_positions[:, None] - det.pmt_positions[None], axis=2)
    np.fill_diagonal(d, np.inf)
    nearest = d.min(axis=1)
    spacing = np.sqrt(4 * np.pi * R_**2 / N_)
    assert nearest.min() > 0.6 * spacing and nearest.max() < 1.4 * spacing
    assert nearest.min() > 2 * r                        # катоды не перекрываются
    assert np.linalg.norm(det.pmt_directions.mean(axis=0)) < 0.05
    octant = (det.pmt_directions > 0) @ np.array([1, 2, 4])
    counts = np.bincount(octant, minlength=8)
    assert np.all(np.abs(counts - N_ / 8) < SIGMA * np.sqrt(N_ / 8))


def test_point_belongs_to_cathode_only_inside_its_edge():
    det = Detector()
    alpha = np.arcsin(r / R)
    for k in range(N):
        n = det.pmt_directions[k]
        t = np.cross(n, [0.3, 0.5, 0.81])
        t /= np.linalg.norm(t)
        inside = R * (np.cos(0.99 * alpha) * n + np.sin(0.99 * alpha) * t)
        outside = R * (np.cos(1.01 * alpha) * n + np.sin(1.01 * alpha) * t)
        assert det.pmt_index(inside)[0] == k
        assert det.pmt_index(outside)[0] == -1
        assert det.pmt_index(-R * n)[0] != k            # противоположная сторона сферы


# --- источник ---

def test_directions_are_isotropic():
    u = Source(rng=101).directions(100.0, fluctuate=False)
    assert np.allclose(np.linalg.norm(u, axis=1), 1)
    assert stats.kstest(u[:, 2], stats.uniform(-1, 2).cdf).pvalue > P_MIN
    assert stats.kstest(np.arctan2(u[:, 1], u[:, 0]), stats.uniform(-np.pi, 2 * np.pi).cdf).pvalue > P_MIN
    n = len(u)
    assert np.all(np.abs(u.mean(axis=0)) < SIGMA / np.sqrt(3 * n))
    second = u.T @ u / n                                # <n_i n_j> = delta_ij / 3
    assert np.all(np.abs(second - np.eye(3) / 3) < SIGMA * 0.3 / np.sqrt(n))


@pytest.mark.parametrize('E', [0.2, 1.0, 3.0])
def test_photon_number_is_poisson_with_mean_YE(E):
    src = Source(rng=102)
    n = np.array([src.n_photons(E) for _ in range(20_000)])
    assert_mean(n, Y * E)
    assert_variance(n, Y * E)
    assert src.n_photons(E, fluctuate=False) == round(Y * E)


def test_generators_give_the_whole_event():
    src = Source(rng=103)
    assert sum(1 for _ in src.photons(0.1, fluctuate=False)) == 1000
    batches = list(src.photon_batches(1.0, fluctuate=False))
    assert [len(b) for b in batches] == [100] * 100
    u = np.concatenate(batches)
    assert np.allclose(np.linalg.norm(u, axis=1), 1)
    assert np.all(np.abs(u.mean(axis=0)) < SIGMA / np.sqrt(3 * len(u)))


# --- трассировка ---

@pytest.mark.parametrize('rho', [0.0, 1.0, 1.7, 1.95])
def test_trace_equals_ray_disc_intersection(rho):
    g = np.random.default_rng(int(rho * 100) + 7)
    det = Detector()
    o = rho * isotropic(g, 1)[0]
    u = isotropic(g, 500_000)
    assert np.array_equal(det.trace(o, u), disc_hits(det.pmt_directions, o, u))


@pytest.mark.parametrize('search', ['brute', 'grid', 'kdtree'])
def test_large_detector_search_equals_ray_disc_intersection(search):
    g = np.random.default_rng(105)
    det = Detector(R=15.0, N=2000, r=r, search=search)
    for rho in (0.0, 10.0):
        o = rho * isotropic(g, 1)[0]
        hits = 0
        for _ in range(20):
            u = isotropic(g, 10_000)
            expected = disc_hits(det.pmt_directions, o, u, R=15.0, r=r)
            assert np.array_equal(det.trace(o, u), expected)
            hits += (expected >= 0).sum()
        assert hits > 2000


def test_path_length_goes_forward_to_the_wall():
    g = np.random.default_rng(106)
    det = Detector()
    o = 1.5 * isotropic(g, 1)[0]
    u = isotropic(g, 100_000)
    forward, backward = det.path_length(o, u), det.path_length(o, -u)
    assert np.all(forward > 0) and np.all(backward > 0)
    assert np.allclose(np.linalg.norm(o + forward[:, None] * u, axis=1), R)
    assert np.allclose(np.linalg.norm(o - backward[:, None] * u, axis=1), R)
    # точка, куда смотрит направление на стенку, ближе противоположной
    toward = o / np.linalg.norm(o)
    assert np.isclose(det.path_length(o, toward[None])[0], R - 1.5)
    assert np.isclose(det.path_length(o, -toward[None])[0], R + 1.5)


# --- эталон сам проверяется по аналитическим формулам ---

def test_reference_integral_matches_analytic_solid_angle():
    n = fibonacci_sphere(N)
    D = np.sqrt(R**2 - r**2)
    centre = disc_fraction(n, np.zeros(3))
    assert np.allclose(centre, (1 - D / R) / 2, rtol=1e-8)
    h = D - 1.7                                         # на оси диска: Omega = 2 pi (1 - h / sqrt(h^2 + r^2))
    on_axis = disc_fraction(n[3:4], 1.7 * n[3])[0]
    assert np.isclose(on_axis, (1 - h / np.hypot(h, r)) / 2, rtol=1e-6)


# --- затухание и доли фотонов ---

@pytest.mark.parametrize('path', [0.5, 2.0, 8.0])
def test_survival_probability(path):
    det = Detector(attenuation_length=L, rng=107)
    alive = det.survives(np.full(1_000_000, path))
    p = np.exp(-path / L)
    assert abs(alive.mean() - p) < SIGMA * np.sqrt(p * (1 - p) / len(alive))


@pytest.mark.parametrize('att', [np.inf, L])
@pytest.mark.parametrize('rho', [0.0, 1.0, 1.7])
def test_trace_fractions_match_integral_on_every_channel(att, rho):
    g = np.random.default_rng(108)
    det = Detector(attenuation_length=att, rng=109)
    o = rho * det.pmt_directions[3]
    counts, n_rays = np.zeros(N), 4_000_000
    for _ in range(40):
        idx = det.trace(o, isotropic(g, n_rays // 40))
        counts += np.bincount(idx[idx >= 0], minlength=N)
    assert_counts(counts, n_rays * disc_fraction(det.pmt_directions, o, att))


@pytest.mark.parametrize('rho, tolerance', [(0.0, 0.003), (1.0, 0.01), (1.4, 0.025), (1.7, 0.09)])
def test_small_disc_approximation_error_is_as_documented(rho, tolerance):
    det = Detector(attenuation_length=L)
    o = rho * det.pmt_directions[3]
    ratio = expected_fraction(det, o) / disc_fraction(det.pmt_directions, o, L)
    assert np.max(np.abs(ratio - 1)) < tolerance


def test_expected_photons_equals_expected_fraction():
    det = Detector(attenuation_length=L)
    pos = np.array([0.5, -0.3, 0.8])
    assert np.allclose(expected_photons(det, pos, 1.7, Y).eval(), Y * 1.7 * expected_fraction(det, pos))


# --- фотоэлектроны ---

def test_photoelectrons_per_photon_are_poisson():
    det = Detector(rng=110)
    pe = det.photoelectrons(np.ones(500_000, dtype=int))
    k = np.arange(5)
    p = stats.poisson.pmf(k, a)
    assert_counts(np.bincount(pe, minlength=8)[:5], len(pe) * p)
    assert abs((pe >= 1).mean() - (1 - np.exp(-a))) < SIGMA * np.sqrt(0.24 / len(pe))


def test_photoelectrons_add_up_over_photons():
    det = Detector(rng=111)
    pe = det.photoelectrons(np.full(200_000, 100))
    assert_mean(pe, a * 100)
    assert_variance(pe, a * 100)
    assert np.all(det.photoelectrons(np.zeros(1000, dtype=int)) == 0)


# --- полный заряд события ---

def test_task_formula_is_four_times_the_area_ratio():
    assert np.isclose(a * Y * N * r**2 / R**2 / (a * Y * N * np.pi * r**2 / (4 * np.pi * R**2)), 4)


@pytest.mark.parametrize('E, z, att', [(0.2, 0.0, np.inf), (1.0, 0.0, np.inf), (3.0, 0.0, np.inf),
                                       (1.0, 1.0, np.inf), (1.0, 0.0, L), (1.0, 1.5, L)])
def test_charge_moments_on_every_channel(E, z, att):
    det = Detector(attenuation_length=att, rng=112)
    origin = np.array([0, 0, z])
    pe = simulate_charge(det, Source(rng=113), E, origin=origin, n_events=1500)
    mu = a * Y * E * disc_fraction(det.pmt_directions, origin, att)
    Q = pe.sum(axis=1)
    assert_mean(Q, mu.sum())
    assert_variance(Q, (1 + a) * mu.sum())              # составное распределение Пуассона
    # среднее каждого канала, дисперсия среднего (1 + a) mu / n_events
    assert_counts(pe.mean(axis=0) * len(pe) / (1 + a), mu * len(pe) / (1 + a))
    corr = np.corrcoef(pe.T)[np.triu_indices(N, 1)]     # каналы независимы (прореживание Пуассона)
    assert abs(corr.mean()) < SIGMA / np.sqrt(len(pe) * len(corr))


@pytest.mark.parametrize('E', [1.0, 3.0])
def test_charge_with_fixed_photon_number(E):
    det = Detector(rng=123)
    Q = simulate_charge(det, Source(rng=124), E, n_events=1500, fluctuate=False).sum(axis=1)
    f = disc_fraction(det.pmt_directions, np.zeros(3)).sum()
    assert_mean(Q, a * Y * E * f)
    assert_variance(Q, a * Y * E * f * (1 + a * (1 - f)))   # биномиальный отбор фотонов


# --- модель PyMC ---

def test_pymc_models_with_fixed_photon_number():
    det = Detector()
    f = disc_fraction(det.pmt_directions, np.zeros(3))
    for model in (photon_model(det, Y, 1.0, fluctuate=False), multinomial_model(det, Y, 1.0, fluctuate=False)):
        s = sample_Q(model, 3000, seed=125)
        assert np.all(s['n_gamma'] == Y)
        assert_counts(s['hits'].sum(axis=0), 3000 * Y * f)
        assert_mean(s['Q'], a * Y * f.sum())
        assert_variance(s['Q'], a * Y * f.sum() * (1 + a * (1 - f.sum())))


@pytest.mark.parametrize('z, att', [(0.0, False), (1.5, True)])
def test_pymc_photon_model_matches_theory(z, att):
    det = Detector(attenuation_length=L if att else np.inf)
    origin = np.array([0, 0, z])
    s = sample_Q(photon_model(det, Y, 1.0, origin, attenuation=att), 3000, seed=114)
    f = disc_fraction(det.pmt_directions, origin, L if att else np.inf)
    assert_mean(s['n_gamma'], Y)
    assert_variance(s['n_gamma'], Y)
    assert_counts(s['hits'].sum(axis=0), 3000 * Y * f)
    assert_mean(s['Q'], a * Y * f.sum())
    assert_variance(s['Q'], (1 + a) * a * Y * f.sum())


def test_pymc_multinomial_model_matches_theory():
    det = Detector()
    s = sample_Q(multinomial_model(det, Y, 1.0), 3000, seed=115)
    f = disc_fraction(det.pmt_directions, np.zeros(3))
    assert_mean(s['n_gamma'], Y)
    assert_counts(s['hits'].sum(axis=0), 3000 * Y * f)
    assert_mean(s['Q'], a * Y * f.sum())
    assert_variance(s['Q'], (1 + a) * a * Y * f.sum())


# --- правдоподобие и приоры задачи 10 ---

@pytest.mark.parametrize('lam', [0.5, 6.0, 25.0])
def test_neyman_logp_equals_direct_convolution(lam):
    m = np.arange(0, 60)
    logp = neyman_logp(m, np.full(len(m), lam), a, 400).eval()
    direct = np.array([neyman_direct(k, lam) for k in m])
    assert np.allclose(logp, direct, rtol=1e-8, atol=1e-8)
    pmf = np.exp(direct)
    assert np.isclose(pmf.sum(), 1, atol=1e-6)
    mean = pmf @ m
    assert np.isclose(mean, a * lam, rtol=1e-4)
    assert np.isclose(pmf @ (m - mean)**2, (1 + a) * a * lam, rtol=1e-3)


@pytest.mark.parametrize('lam', [0.5, 6.0, 25.0])
def test_neyman_random_follows_the_direct_pmf(lam):
    s = neyman_random(np.full(500_000, lam), a, rng=np.random.default_rng(116), size=500_000)
    pmf = np.exp([neyman_direct(k, lam) for k in range(80)])
    keep = len(s) * pmf > 20
    assert_counts(np.bincount(s, minlength=80)[:80][keep], len(s) * pmf[keep])


def test_model_likelihood_equals_direct_sum():
    det = Detector(attenuation_length=L, rng=117)
    pe = det.response(np.array([0.5, -0.3, 0.8]), Source(rng=118).directions(1.5))
    model = reconstruction_model(det, Y, pe)
    fn = model.compile_fn(model.logp(vars=model.observed_RVs))
    h_max = 2 * pe.max() / a + 40
    g = np.random.default_rng(119)
    D = np.sqrt(R**2 - r**2)

    def compare(E, s):
        # параметры задаём сами: RV из графа модели при вызове дали бы случайный розыгрыш приора
        v = g.normal(size=3)
        logp = fn({'E_interval__': logit(E / 10.0), 's_logodds__': logit(s), 'v': v})
        to_disc = D * det.pmt_directions - 0.85 * R * s * v / np.linalg.norm(v)
        d = np.linalg.norm(to_disc, axis=1)
        cos = np.sum(to_disc * det.pmt_directions, axis=1) / d
        lam = Y * E * r**2 * cos / (4 * d**2) * np.exp(-d / L)
        return logp, sum(neyman_direct(m, l) for m, l in zip(pe, lam)), lam.max()

    checked = 0
    for _ in range(12):
        logp, direct, lam_max = compare(g.uniform(0.5, 3.0), g.uniform(0.1, 0.85))
        if lam_max < 0.8 * h_max:                       # усечение суммы здесь не сказывается
            assert np.isclose(logp, direct, rtol=1e-8, atol=1e-8)
            checked += 1
    assert checked >= 6
    # при больших lambda усечение может только занизить правдоподобие
    logp, direct, lam_max = compare(9.0, 0.85)
    assert lam_max > h_max and logp <= direct + 1e-9


def test_priors_are_flat_in_energy_and_uniform_in_the_ball():
    det = Detector(attenuation_length=L)
    model = reconstruction_model(det, Y, np.zeros(N, dtype=int))
    with model:
        prior = pm.sample_prior_predictive(5000, var_names=['E', 'x', 'y', 'z', 'rho'], random_seed=120).prior
    E, x, y, z, rho = [prior[k].values.ravel() for k in ('E', 'x', 'y', 'z', 'rho')]
    assert np.allclose(np.sqrt(x**2 + y**2 + z**2), rho)
    assert stats.kstest(E / 10.0, 'uniform').pvalue > P_MIN
    assert stats.kstest((rho / (0.85 * R))**3, 'uniform').pvalue > P_MIN
    assert stats.kstest(z / rho, stats.uniform(-1, 2).cdf).pvalue > P_MIN


# --- ноутбуки ---

def shared_seeds(text):
    """Seed, которые в одном тексте получили и Detector, и Source."""
    det = set(re.findall(r'Detector\([^)\n]*rng=(\d+)', text))
    src = set(re.findall(r'Source\([^)\n]*rng=(\d+)', text))
    return det & src


def test_shared_seed_detector_finds_a_shared_seed():
    assert shared_seeds('d = Detector(R=2.0, rng=6)\ns = Source(Y=1, rng=6)') == {'6'}
    assert shared_seeds('d = Detector(R=2.0, rng=6)\ns = Source(Y=1, rng=60)') == set()


@pytest.mark.parametrize('path', sorted(glob.glob(str(HERE / 'task*.ipynb'))))
def test_notebooks_are_executed_cleanly_with_separate_seeds(path):
    nb = nbformat.read(path, as_version=4)
    code = [c for c in nb.cells if c.cell_type == 'code']
    assert [c.execution_count for c in code] == list(range(1, len(code) + 1))
    assert not [o for c in code for o in c.outputs if o.output_type == 'error']
    assert not shared_seeds('\n'.join(c.source for c in code))
    # сохранённая модель виджета без связи с ядром блокирует загрузку всех новых виджетов k3d
    assert 'widgets' not in nb.metadata
    assert not [o for c in code for o in c.outputs if 'application/vnd.jupyter.widget-view+json' in o.get('data', {})]


# --- калибровка реконструкции (несколько минут: pytest -m slow) ---

@pytest.mark.slow
def test_reconstruction_pulls_are_standard_normal():
    det = Detector(attenuation_length=L, rng=121)
    g = np.random.default_rng(122)
    E_true, n_events = 2.0, 24
    pulls = []
    for i in range(n_events):
        v = g.normal(size=3)
        xt = v / np.linalg.norm(v) * 1.5 * g.uniform()**(1 / 3)
        pe = det.response(xt, Source(rng=40_000 + i).directions(E_true))
        _, idata = reconstruct(det, Y, pe, draws=500, tune=500, chains=2, seed=i,
                               compute_convergence_checks=False)
        post = idata.posterior
        truth = dict(E=E_true, x=xt[0], y=xt[1], z=xt[2])
        pulls.append([(post[k].values.mean() - truth[k]) / post[k].values.std() for k in 'Exyz'])
    pulls = np.array(pulls)
    assert np.all(np.abs(pulls.mean(axis=0)) < SIGMA / np.sqrt(n_events)), pulls.mean(axis=0)
    assert np.all(np.abs(pulls.std(axis=0) - 1) < SIGMA / np.sqrt(2 * n_events)), pulls.std(axis=0)
