"""Проверка самих проверок: в модель вносятся заведомые ошибки, test_physics.py должен их поймать.

Каждая ошибка вносится в отдельном процессе подменой функции; файлы проекта не меняются.
Запуск: python check_mutations.py         все ошибки, кроме медленной
        python check_mutations.py --slow  ещё и ошибка, которую ловит только калибровка
"""
import concurrent.futures
import os
import re
import subprocess
import sys

import numpy as np
import pymc as pm
import pytensor.tensor as pt

import detector
import pymc_model
import reconstruction as rc
import simulation
import source
from detector import Detector
from source import Source


def brute_with(condition):
    def pmt_index_brute(self, points):
        points = np.atleast_2d(points)
        cos = points @ self.pmt_directions.T
        rho2 = np.sum(points**2, axis=1, keepdims=True) - cos**2
        hit = condition(rho2, cos, self.r)
        return np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
    Detector.pmt_index_brute = pmt_index_brute


def isotropic_with(make):
    def _isotropic(self, n):
        phi, cos = make(self.rng, n)
        sin = np.sqrt(1 - cos**2)
        return np.stack([sin * np.cos(phi), sin * np.sin(phi), cos], axis=1)
    Source._isotropic = _isotropic


def survives_with(probability):
    def survives(self, path):
        if np.isinf(self.attenuation_length):
            return np.ones(len(path), dtype=bool)
        return self.rng.uniform(size=len(path)) < probability(self, path)
    Detector.survives = survives


def wrong_r():
    init = Detector.__init__
    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self.r *= 1.03
    Detector.__init__ = __init__


def clumped_pmts():
    def fibonacci_sphere(N):
        i = np.arange(N) + 0.5
        cos = 1 - i / N                                # только верхняя полусфера
        sin = np.sqrt(1 - cos**2)
        phi = np.pi * (3 - np.sqrt(5)) * i
        return np.stack([sin * np.cos(phi), sin * np.sin(phi), cos], axis=1)
    detector.fibonacci_sphere = fibonacci_sphere


def backward_root():
    def path_length(self, origins, directions):
        origins = np.broadcast_to(origins, directions.shape)
        b = np.sum(origins * directions, axis=1)
        c = np.sum(origins**2, axis=1) - self.R**2
        return b + np.sqrt(b**2 - c)                   # путь против направления луча
    Detector.path_length = path_length


def rolled_channels():
    response = Detector.response
    Detector.response = lambda self, o, d: np.roll(response(self, o, d), 1)


def lossy_search(name):
    search = getattr(Detector, name)
    def lossy(self, points):
        index = search(self, points)
        index[::50] = -1
        return index
    setattr(Detector, name, lossy)


def kdtree_without_check():
    def pmt_index_kdtree(self, points):
        from scipy.spatial import cKDTree
        tree = cKDTree(self.pmt_directions)
        _, k = tree.query(points / np.linalg.norm(points, axis=1, keepdims=True))
        return k
    Detector.pmt_index_kdtree = pmt_index_kdtree


def neyman_with_a_squared():
    logp = rc.neyman_logp
    rc.neyman_logp = lambda pe, lam, a, h_max: logp(pe, lam, a**2, h_max)


def neyman_is_poisson():
    def logp(pe, lam, a, h_max):
        return pe * pt.log(a * lam) - a * lam - pt.gammaln(pe + 1)
    rc.neyman_logp = logp


def lam_scaled(factor):
    photons = rc.expected_photons
    rc.expected_photons = lambda det, pos, E, Y: factor * photons(det, pos, E, Y)


def lam_variant(use_cos, use_attenuation):
    def expected_photons(det, pos, E, Y):
        v = pt.constant(np.sqrt(det.R**2 - det.r**2) * det.pmt_directions) - pos
        d = pt.sqrt(pt.sum(v**2, axis=1))
        frac = det.r**2 / (4 * d**2)
        if use_cos:
            frac = frac * pt.sum(v * det.pmt_directions, axis=1) / d
        if use_attenuation and np.isfinite(det.attenuation_length):
            frac = frac * pt.exp(-d / det.attenuation_length)
        return Y * E * frac
    rc.expected_photons = expected_photons


def prior_wrong_radius():
    beta = pm.Beta
    pm.Beta = lambda name, alpha, b, **kwargs: beta(name, 2, b, **kwargs)


def prior_wrong_energy():
    uniform = pm.Uniform
    def wrapped(name, lower=0.0, upper=1.0, **kwargs):
        return uniform(name, lower, upper / 2 if name == 'E' else upper, **kwargs)
    pm.Uniform = wrapped


def pymc_with_detector(**changes):
    model = pymc_model.photon_model
    def photon_model(det, Y, E, origin=(0, 0, 0), attenuation=False, n_max=None):
        params = dict(R=det.R, N=det.N, r=det.r, a=det.a, attenuation_length=det.attenuation_length)
        return model(Detector(**{**params, **changes}), Y, E, origin, attenuation, n_max)
    pymc_model.photon_model = photon_model


def pymc_without_attenuation():
    model = pymc_model.photon_model
    pymc_model.photon_model = lambda det, Y, E, origin=(0, 0, 0), attenuation=False, n_max=None: model(det, Y, E, origin, False, n_max)


def multinomial_wrong_r():
    model = pymc_model.multinomial_model
    def multinomial_model(det, Y, E, origin=(0, 0, 0), attenuation=False):
        return model(Detector(R=det.R, N=det.N, r=det.r * 1.05, a=det.a), Y, E, origin, attenuation)
    pymc_model.multinomial_model = multinomial_model


def fraction_times_4():
    fraction = simulation.expected_fraction
    simulation.expected_fraction = lambda det, origin=(0, 0, 0): 4 * fraction(det, origin)


def set_attr(owner, name, value):
    return lambda: setattr(owner, name, value)


MUTANTS = {
    'контроль: без ошибки': lambda: None,
    # геометрия и трассировка
    'ФЭУ только на верхней полусфере': clumped_pmts,
    'радиус катода больше на 3%': wrong_r,
    'катод шире: r^2 * 1.1': lambda: brute_with(lambda rho2, cos, r: (rho2 <= 1.1 * r**2) & (cos > 0)),
    'нет проверки cos > 0 (катод с другой стороны сферы)': lambda: brute_with(lambda rho2, cos, r: rho2 <= r**2),
    'путь до стенки против луча': backward_root,
    'сетка теряет 2% точек': lambda: lossy_search('pmt_index_grid'),
    'k-d дерево без проверки катода': kdtree_without_check,
    'заряд записан в соседний канал': rolled_channels,
    # источник
    'равномерно по theta вместо cos(theta)': lambda: isotropic_with(lambda g, n: (g.uniform(0, 2 * np.pi, n), np.cos(g.uniform(0, np.pi, n)))),
    'phi только до pi': lambda: isotropic_with(lambda g, n: (g.uniform(0, np.pi, n), g.uniform(-1, 1, n))),
    'число фотонов не случайно': set_attr(Source, 'n_photons', lambda self, E, fluctuate=True: int(round(self.Y * E))),
    'выход нелинеен по энергии': set_attr(Source, 'n_photons', lambda self, E, fluctuate=True: int(self.rng.poisson(self.Y * E**1.05))),
    # затухание
    'длина затухания вдвое больше': lambda: survives_with(lambda self, path: np.exp(-path / (2 * self.attenuation_length))),
    'путь всегда равен R': lambda: survives_with(lambda self, path: np.full(len(path), np.exp(-self.R / self.attenuation_length))),
    # фотоэлектроны
    'a не учитывается': set_attr(Detector, 'photoelectrons', lambda self, n: self.rng.poisson(np.asarray(n))),
    'биномиальные ф.э. вместо пуассоновских': set_attr(Detector, 'photoelectrons', lambda self, n: self.rng.binomial(np.asarray(n), self.a)),
    # аналитическая доля
    'доля в expected_fraction больше в 4 раза': fraction_times_4,
    # модель PyMC
    'PyMC: другое a': lambda: pymc_with_detector(a=0.55),
    'PyMC: другой r': lambda: pymc_with_detector(r=0.105),
    'PyMC: затухание выключено': pymc_without_attenuation,
    'PyMC: другой r в мультиномиальной модели': multinomial_wrong_r,
    # правдоподобие и приоры
    'Нейман: a^2 вместо a': neyman_with_a_squared,
    'Нейман заменён на Пуассон': neyman_is_poisson,
    'розыгрыш без числа фотонов': set_attr(rc, 'neyman_random', lambda lam, a, rng=None, size=None: rng.poisson(a * lam, size=size)),
    'lambda меньше на 10%': lambda: lam_scaled(0.9),
    'в lambda нет cos(psi)': lambda: lam_variant(False, True),
    'в lambda нет затухания': lambda: lam_variant(True, False),
    'приор по радиусу: Beta(2, 1)': prior_wrong_radius,
    'приор по энергии до 5 МэВ': prior_wrong_energy,
}
SLOW = {'калибровка: контроль без ошибки': lambda: None,
        'калибровка: lambda меньше на 20%': lambda: lam_scaled(0.8)}


def run_one(name):
    """Выполняется в дочернем процессе: внести ошибку и запустить тесты."""
    import pytest
    if name in SLOW:
        SLOW[name]()
        return pytest.main(['test_physics.py', '-x', '-q', '-m', 'slow', '-p', 'no:cacheprovider'])
    MUTANTS[name]()
    return pytest.main(['test_physics.py', '-x', '-q', '-p', 'no:cacheprovider'])


def launch(name):
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
    done = subprocess.run([sys.executable, __file__, '--one', name], capture_output=True, encoding='utf-8', env=env)
    failed = re.findall(r'^FAILED test_physics\.py::(\S+)', done.stdout, flags=re.M)
    return name, done.returncode, failed[0] if failed else ''


def main(names):
    workers = max(2, (os.cpu_count() or 4) // 2)
    survived = 0
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for name, code, failed in pool.map(launch, names):
            control = 'контроль' in name
            ok = (code == 0) if control else (code == 1)
            survived += not ok
            verdict = ('прошёл' if code == 0 else f'УПАЛ: {failed}') if control else (f'поймана: {failed}' if code == 1 else f'НЕ ПОЙМАНА (код {code})')
            print(f"{'ok ' if ok else 'ПЛОХО'}  {name:55s}  {verdict}", flush=True)
    print(f'итог: проверено {len(names)}, неверных исходов: {survived}')
    return 1 if survived else 0


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    if '--one' in sys.argv:
        sys.exit(run_one(sys.argv[sys.argv.index('--one') + 1]))
    sys.exit(main(list(SLOW) if '--slow' in sys.argv else list(MUTANTS)))
