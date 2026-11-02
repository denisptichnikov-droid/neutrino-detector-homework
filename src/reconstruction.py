"""Байесовская реконструкция энергии и вершины события по зарядам ФЭУ.

Правдоподобие. Фотоны рождаются по Пуассону со средним Y E и независимо
попадают на катод k с вероятностью p_k(x) (телесный угол и затухание).
Прореживание пуассоновского потока даёт независимые числа фотонов на катодах:
    hits_k ~ Poisson(lambda_k),   lambda_k = Y E p_k(x).
Каждый фотон даёт Poisson(a) ф.э., поэтому при заданном hits_k
    pe_k | hits_k ~ Poisson(a hits_k),
а после суммирования по hits_k -- распределение Неймана типа A:
    P(pe_k = m) = sum_h Poisson(h; lambda_k) Poisson(m; a h).
Сумма по h считается явно (до h_max), поэтому правдоподобие точное и
дифференцируемое по lambda_k, а значит, по E и x -- подходит для NUTS.
Усечение занижает правдоподобие только при lambda_k >~ h_max, т.е. там, где оно и так
мало: на сэмплах постериора |d log P| < 1e-12.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from detector import Detector


def expected_photons(det: Detector, pos, E, Y):
    """lambda_k: среднее число фотонов на катоде k от события (E, pos), pytensor.

    Приближение малого диска, как `simulation.expected_fraction`:
    Omega_k / 4 pi = r^2 cos(psi_k) / (4 d_k^2), затухание exp(-d_k / L).
    Точность падает к стенке: при |x| = 1.7 м под катодом доля на этом канале завышена
    на 8.5%. На событиях 3 МэВ в 1.6 м под катодом замена на точный интеграл по диску
    не изменила восстановленные E и |x| в пределах статистики.
    """
    centers = pt.constant(np.sqrt(det.R**2 - det.r**2) * det.pmt_directions)
    v = centers - pos
    d = pt.sqrt(pt.sum(v**2, axis=1))
    cos_psi = pt.sum(v * det.pmt_directions, axis=1) / d
    frac = det.r**2 * cos_psi / (4 * d**2)
    if np.isfinite(det.attenuation_length):
        frac = frac * pt.exp(-d / det.attenuation_length)
    return Y * E * frac


def neyman_logp(pe, lam, a, h_max):
    """log P(pe_k) для распределения Неймана типа A, массив по каналам."""
    h = pt.arange(h_max)[:, None]                                    # (H, 1)
    log_hits = h * pt.log(lam) - lam - pt.gammaln(h + 1)             # (H, N)
    mu = a * h
    log_pe = pt.switch(pt.eq(h, 0),
                       pt.switch(pt.eq(pe, 0), 0.0, -np.inf),
                       pe * pt.log(pt.maximum(mu, 1e-300)) - mu - pt.gammaln(pe + 1))
    return pt.logsumexp(log_hits + log_pe, axis=0)


def neyman_random(lam, a, rng=None, size=None):
    """Розыгрыш pe_k: hits_k ~ Poisson(lambda_k), затем pe_k ~ Poisson(a hits_k)."""
    return rng.poisson(a * rng.poisson(lam, size=size))


def reconstruction_model(det: Detector, Y: float, pe_obs: np.ndarray,
                         E_max: float = 10.0, r_max: float = None,
                         likelihood: str = 'neyman') -> pm.Model:
    """Модель для вывода (E, x, y, z) по наблюдённым ф.э. pe_obs (массив (N,)).

    Приоры: E ~ U(0, E_max); вершина равномерна в шаре радиуса r_max
    (фидуциальный объём, по умолчанию 0.85 R: у самой стенки приближение
    малого диска грубое, а правдоподобие очень крутое -- NUTS там
    расходится и застревает).
    Равномерность в шаре задаётся без жёсткой стенки (о неё NUTS «ломается» --
    расходящиеся траектории): направление -- нормированный гауссов вектор
    (изотропен), относительный радиус s ~ Beta(3, 1) с плотностью 3 s^2 --
    ровно распределение |x| / r_max для точки, равномерной в шаре.
    likelihood='neyman' -- точное правдоподобие, 'poisson' -- приближение
    pe_k ~ Poisson(a lambda_k), игнорирующее флуктуации числа ф.э. на фотон.
    """
    if r_max is None:
        r_max = 0.85 * det.R
    pe_obs = np.asarray(pe_obs)
    with pm.Model() as model:
        E = pm.Uniform('E', 0, E_max)
        v = pm.Normal('v', 0, 1, shape=3)
        s = pm.Beta('s', 3, 1)
        rho = pm.Deterministic('rho', r_max * s)
        pos = rho * v / pt.sqrt(pt.sum(v**2))
        x = pm.Deterministic('x', pos[0])
        y = pm.Deterministic('y', pos[1])
        z = pm.Deterministic('z', pos[2])

        lam = pm.Deterministic('lam', expected_photons(det, pos, E, Y))
        if likelihood == 'neyman':
            h_max = int(2 * pe_obs.max() / det.a + 40)
            pm.CustomDist('pe', lam,
                          logp=lambda value, mu: neyman_logp(value, mu, det.a, h_max),
                          random=lambda mu, rng=None, size=None: neyman_random(mu, det.a, rng, size),
                          observed=pe_obs)
        else:
            pm.Poisson('pe', mu=det.a * lam, observed=pe_obs)
    model.r_max = r_max
    model.E_max = E_max
    return model


def start_point(det: Detector, Y: float, pe_obs: np.ndarray, model: pm.Model):
    """Начальная точка для цепей: максимум апостериорной плотности (MAP).

    У стенки детектора есть локальные максимумы (вершина вплотную к
    какому-нибудь ФЭУ), и цепь, стартовавшая рядом, может там застрять:
    правдоподобие в них на ~25 единиц log ниже главного, но NUTS не может
    перепрыгнуть между ними. Поэтому MAP ищется из нескольких стартов --
    центр детектора и точки вдоль «центра тяжести» зарядов
    u = sum(pe_k n_k) / sum(pe_k), -- и берётся лучший.
    """
    from simulation import expected_fraction
    pe_obs = np.asarray(pe_obs)
    Q = max(pe_obs.sum(), 1)
    u = (pe_obs @ det.pmt_directions) / Q
    u_dir = u / max(np.linalg.norm(u), 1e-9)
    r_max = model.r_max
    best, best_lp = None, -np.inf
    logp = model.compile_logp()
    for frac in (0.0, 0.25, 0.5, 0.75):
        pos = frac * r_max * u_dir
        E0 = Q / (det.a * Y * expected_fraction(det, pos).sum())
        v0 = u_dir if frac > 0 else np.array([0.0, 0.0, 1.0])
        start = {'E': np.clip(E0, 0.05, 0.95 * model.E_max),
                 'v': v0, 's': max(frac, 0.05)}
        mp = pm.find_MAP(start=start, model=model, progressbar=False)
        point = {**model.initial_point(), **{k: mp[k] for k in model.initial_point()}}
        lp = logp(point)
        if lp > best_lp:
            best, best_lp = mp, lp
    return {'E': best['E'], 'v': best['v'], 's': best['s']}


def reconstruct(det: Detector, Y: float, pe_obs: np.ndarray, draws=1000, tune=1000,
                chains=4, seed=None, likelihood='neyman', **kwargs):
    """Постериор (E, x, y, z) для одного события: MAP-старт + NUTS."""
    model = reconstruction_model(det, Y, pe_obs, likelihood=likelihood)
    init = start_point(det, Y, pe_obs, model)
    with model:
        idata = pm.sample(draws, tune=tune, chains=chains, cores=1, random_seed=seed,
                          initvals=init, progressbar=False, target_accept=0.9, **kwargs)
    return model, idata
