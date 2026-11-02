"""Модель детектора (задачи 1-3, 6) в виде вероятностного графа PyMC.

Геометрия (координаты ФЭУ) берётся из `Detector` -- в графе она константа.
Всё, что разыгрывается при событии, -- узлы графа:

    n_gamma = Y E или ~ Poisson(Y E)             число фотонов (задачи 2, 5)
    phi, cosTheta ~ Uniform                      направления фотонов (задача 2)
    t = путь до стенки, p = o + t n              пропагация (задача 3)
    on_k = точка p на катоде k                   попадание в ФЭУ (задачи 1, 3)
    u ~ Uniform(0, 1), u < exp(-t/L)             затухание (задача 6, опционально)
    hits_k = число фотонов на катоде k           Deterministic
    pe_k ~ Poisson(a hits_k)                     фотоэлектроны (задача 3)
    Q = sum_k pe_k                               полный заряд
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from detector import Detector


def polar_to_cartesian(phi, cosTheta):
    sinTheta = pt.sqrt(1 - cosTheta**2)
    return pt.stack([sinTheta * pt.cos(phi),
                     sinTheta * pt.sin(phi),
                     cosTheta], axis=-1)


def photon_counter(Y: float, E: float, fluctuate: bool):
    """Узел числа фотонов: ровно Y E (задачи 1-3) или Poisson(Y E) (задача 5)."""
    if fluctuate:
        return pm.Poisson('n_gamma', mu=Y * E)
    return pm.Deterministic('n_gamma', pt.constant(int(round(Y * E))))


def photon_model(det: Detector, Y: float, E: float, origin=(0, 0, 0),
                 attenuation: bool = False, n_max: int = None,
                 fluctuate: bool = True) -> pm.Model:
    """Пофотонная модель: каждый фотон -- отдельный элемент массива в графе.

    Размер массивов в графе должен быть фиксирован, а число фотонов случайно.
    Поэтому направления разыгрываются для n_max фотонов, а учитываются
    только первые n_gamma (маска `emitted`). n_max = <n> + 7 sigma: вероятность
    n_gamma > n_max ~ 1e-12, так что обрезка не влияет на результат.
    """
    mean = Y * E
    if n_max is None:
        n_max = int(np.ceil(mean + 7 * np.sqrt(mean)))
    o = pt.constant(np.asarray(origin, dtype=float))
    pmt_dirs = pt.constant(det.pmt_directions)

    with pm.Model() as model:
        n_gamma = photon_counter(Y, E, fluctuate)
        phi = pm.Uniform('phi', 0, 2 * np.pi, shape=n_max)
        cosTheta = pm.Uniform('cosTheta', -1, 1, shape=n_max)
        n = polar_to_cartesian(phi, cosTheta)                        # (n_max, 3)
        emitted = pt.lt(pt.arange(n_max), n_gamma)                   # первые n_gamma

        # путь до стенки: положительный корень |o + t n|^2 = R^2
        b = pt.dot(n, o)
        t = -b + pt.sqrt(b**2 - pt.sum(o**2) + det.R**2)
        p = o + t[:, None] * n                                       # точка на сфере

        # точка на катоде k: p·n_k > 0 и |p_T|^2 = R^2 - (p·n_k)^2 <= r^2
        cos = pt.dot(p, pmt_dirs.T)                                  # (n_max, N)
        on = pt.and_(pt.gt(cos, 0), pt.le(det.R**2 - cos**2, det.r**2))

        alive = emitted
        if attenuation:
            u = pm.Uniform('u', 0, 1, shape=n_max)
            alive = pt.and_(alive, pt.lt(u, pt.exp(-t / det.attenuation_length)))

        hits = pm.Deterministic('hits', pt.sum(pt.and_(on, alive[:, None]), axis=0))
        pe = pm.Poisson('pe', mu=det.a * hits, shape=det.N)
        pm.Deterministic('Q', pt.sum(pe))
    return model


def multinomial_model(det: Detector, Y: float, E: float, origin=(0, 0, 0),
                      attenuation: bool = False, fluctuate: bool = True) -> pm.Model:
    """Компактная модель: распределение фотонов по ФЭУ одним мультиномиальным узлом.

    Каждый фотон независимо попадает на катод k с вероятностью p_k (телесный
    угол катода / 4 pi, умноженный на вероятность выжить на пути d_k), либо
    никуда не попадает. Тогда числа фотонов на катодах -- Multinomial(n_gamma, p).
    Это то же распределение, что даёт пофотонная модель, но без 10^4 переменных.
    p_k -- приближение малого диска (как `simulation.expected_fraction`),
    записанное через pytensor.
    """
    o = pt.constant(np.asarray(origin, dtype=float))
    centers = pt.constant(np.sqrt(det.R**2 - det.r**2) * det.pmt_directions)
    v = centers - o
    d = pt.sqrt(pt.sum(v**2, axis=1))
    cos_psi = pt.sum(v * det.pmt_directions, axis=1) / d
    frac = det.r**2 * cos_psi / (4 * d**2)
    if attenuation:
        frac = frac * pt.exp(-d / det.attenuation_length)

    with pm.Model() as model:
        p = pm.Deterministic('p', frac)
        n_gamma = photon_counter(Y, E, fluctuate)
        hits_all = pm.Multinomial('hits_all', n=n_gamma,
                                  p=pt.concatenate([p, [1 - pt.sum(p)]]))
        hits = pm.Deterministic('hits', hits_all[:det.N])
        pe = pm.Poisson('pe', mu=det.a * hits, shape=det.N)
        pm.Deterministic('Q', pt.sum(pe))
    return model


def sample_Q(model: pm.Model, draws: int, seed=None) -> dict:
    """Прямое (предковое) сэмплирование графа: draws независимых событий.

    Возвращает словарь массивов 'Q', 'hits', 'n_gamma' с первой осью по событиям.
    """
    with model:
        idata = pm.sample_prior_predictive(draws, var_names=['Q', 'hits', 'n_gamma'],
                                           random_seed=seed)
    return {k: idata.prior[k].values.reshape(draws, *idata.prior[k].shape[2:])
            for k in ('Q', 'hits', 'n_gamma')}
