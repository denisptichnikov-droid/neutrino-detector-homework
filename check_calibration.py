"""Отдельные проверки, на которые ссылаются выводы задач 3, 4 и 10. Счёт занимает около полутора часов.

Запуск: python check_calibration.py
1. Заряд на большой статистике: центр и z = 1 м.
2. Серии реконструкций с новыми seed; несошедшиеся события пересчитываются длинными цепями.
3. Вершина под катодом: приближение малого диска против точного интеграла по диску.
4. Зачем нужны MAP-старт, приор без стенки и фидуциальный объём: что будет без них.
"""
import logging
import sys
import warnings

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from detector import Detector
from source import Source
from simulation import simulate_charge
import reconstruction as rc

logging.getLogger("pytensor.configdefaults").setLevel(logging.ERROR)
logging.getLogger("pymc").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

Y = 10_000


def charge_high_statistics(n=100_000):
    """Фиксированное число фотонов, как в задачах 3 и 4."""
    result = {}
    for z, seed_det, seed_src in ((0.0, 7001, 7002), (1.0, 7003, 7004)):
        Q = simulate_charge(Detector(rng=seed_det), Source(rng=seed_src), 1.0, origin=(0, 0, z),
                            n_events=n, fluctuate=False).sum(axis=1)
        result[z] = Q
        print(f"z = {z} м: <Q> = {Q.mean():.3f} ± {Q.std() / np.sqrt(n):.3f}, sigma = {Q.std():.3f}", flush=True)
    change = result[1.0].mean() / result[0.0].mean() - 1
    error = np.hypot(*[q.std() / q.mean() for q in result.values()]) / np.sqrt(n)
    print(f"изменение при смещении на 1 м: {change:+.3%} ± {error:.3%}", flush=True)


def fit_summary(idata, truth):
    post = {k: idata.posterior[k].values.ravel() for k in 'Exyz'}
    rhat = az.rhat(idata, var_names=list('Exyz'))
    return dict(pulls=[(post[k].mean() - truth[k]) / post[k].std() for k in 'Exyz'],
                E=post['E'].mean(), sigma_E=post['E'].std(),
                radius=np.sqrt(post['x']**2 + post['y']**2 + post['z']**2).mean(),
                rhat=max(float(rhat[k]) for k in 'Exyz'),
                divergent=float(idata.sample_stats.diverging.mean()))


def random_vertex(g, r_max):
    v = g.normal(size=3)
    return v / np.linalg.norm(v) * r_max * g.uniform()**(1 / 3)


def series(E, r_max, n, seed_base):
    det = Detector(attenuation_length=8.0, rng=seed_base + 1)
    g = np.random.default_rng(seed_base + 2)
    rows = []
    for i in range(n):
        xt = random_vertex(g, r_max)
        pe = det.response(xt, Source(rng=seed_base + 1000 + i).directions(E))
        truth = dict(E=E, x=xt[0], y=xt[1], z=xt[2])
        _, idata = rc.reconstruct(det, Y, pe, draws=500, tune=500, chains=2, seed=i, compute_convergence_checks=False)
        row = fit_summary(idata, truth)
        if row['rhat'] > 1.03 or row['divergent'] > 0.01:
            _, idata = rc.reconstruct(det, Y, pe, draws=2000, tune=2000, chains=4, seed=i, compute_convergence_checks=False)
            longer = fit_summary(idata, truth)
            print(f"  длинные цепи: r_hat {row['rhat']:.3f} -> {longer['rhat']:.3f}, расхождений {row['divergent']:.1%} -> {longer['divergent']:.1%}, "
                  f"E {row['E']:.3f} -> {longer['E']:.3f}", flush=True)
            row = longer
        rows.append(row)
    pulls = np.array([r['pulls'] for r in rows])
    mean_E = np.array([r['E'] for r in rows])
    text = '; '.join(f"{k}: {pulls[:, j].mean():+.2f} ± {pulls[:, j].std():.2f}" for j, k in enumerate('Exyz'))
    print(f"серия E = {E} МэВ, радиус до {r_max} м, {n} событий: E = {mean_E.mean():.3f} ± {mean_E.std() / np.sqrt(n):.3f}; "
          f"пуллы {text}; ошибка среднего пулла {1 / np.sqrt(n):.2f}; итоговый max r_hat {max(r['rhat'] for r in rows):.3f}; "
          f"max доля расхождений {max(r['divergent'] for r in rows):.1%}", flush=True)


def exact_photons(det, pos, E, Y):
    """lambda_k как интеграл по диску: 8 узлов квадратуры вместо центра диска."""
    D = np.sqrt(det.R**2 - det.r**2)
    x, w = np.polynomial.legendre.leggauss(2)
    s, w = 0.5 * (x + 1), 0.5 * w
    phi = (np.arange(4) + 0.5) * 2 * np.pi / 4
    u = (np.sqrt(s)[:, None] * np.cos(phi)[None]).ravel()
    v = (np.sqrt(s)[:, None] * np.sin(phi)[None]).ravel()
    weight = (w[:, None] / 4 * np.ones(4)[None]).ravel()
    n = det.pmt_directions
    axis = np.where(np.abs(n[:, 2:]) < 0.9, [[0, 0, 1.0]], [[1.0, 0, 0]])
    e1 = np.cross(n, axis)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(n, e1)
    nodes = pt.constant(D * n[:, None] + det.r * (u[None, :, None] * e1[:, None] + v[None, :, None] * e2[:, None]))
    to_node = nodes - pos
    d = pt.sqrt(pt.sum(to_node**2, axis=2))
    cos = pt.sum(to_node * pt.constant(n[:, None]), axis=2) / d
    frac = det.r**2 / 4 * cos / d**2 * pt.exp(-d / det.attenuation_length)
    return Y * E * pt.sum(pt.constant(weight) * frac, axis=1)


def under_cathode(E=3.0, n=16):
    det = Detector(attenuation_length=8.0, rng=8101)
    g = np.random.default_rng(8102)
    events = []
    for i in range(n):
        xt = 1.6 * det.pmt_directions[g.integers(0, det.N)] + 0.02 * g.normal(size=3)
        events.append((xt, det.response(xt, Source(rng=8200 + i).directions(E))))
    small_disc = rc.expected_photons
    for name, photons in (('малый диск', small_disc), ('интеграл по диску', exact_photons)):
        rc.expected_photons = photons
        rows = []
        for j, (xt, pe) in enumerate(events):
            _, idata = rc.reconstruct(det, Y, pe, draws=500, tune=600, chains=2, seed=j, compute_convergence_checks=False)
            rows.append(fit_summary(idata, dict(E=E, x=xt[0], y=xt[1], z=xt[2])))
        mean_E = np.array([r['E'] for r in rows])
        shift = np.array([r['radius'] - np.linalg.norm(xt) for r, (xt, _) in zip(rows, events)])
        print(f"под катодом, {name}: E = {mean_E.mean():.3f} ± {mean_E.std() / np.sqrt(n):.3f}; "
              f"смещение радиуса {shift.mean() * 100:+.1f} ± {shift.std() / np.sqrt(n) * 100:.1f} см", flush=True)
    rc.expected_photons = small_disc


def report_chains(name, rows):
    print(f"{name}: событий с r_hat > 1.05: {sum(r['rhat'] > 1.05 for r in rows)} из {len(rows)}, max r_hat {max(r['rhat'] for r in rows):.2f}; "
          f"средняя E {np.mean([r['E'] for r in rows]):.3f}; медианная доля расхождений {np.median([r['divergent'] for r in rows]):.1%}, "
          f"максимум {max(r['divergent'] for r in rows):.1%}", flush=True)


def without_map_start(E=1.5, n=40):
    """Как `rc.reconstruct`, но цепи стартуют из начальной точки PyMC, а не из MAP."""
    det = Detector(attenuation_length=8.0, rng=8301)
    g = np.random.default_rng(8302)
    rows = []
    for i in range(n):
        xt = random_vertex(g, 1.5)
        pe = det.response(xt, Source(rng=8400 + i).directions(E))
        with rc.reconstruction_model(det, Y, pe):
            idata = pm.sample(500, tune=500, chains=2, cores=1, random_seed=i, progressbar=False, target_accept=0.9,
                              compute_convergence_checks=False)
        rows.append(fit_summary(idata, dict(E=E, x=xt[0], y=xt[1], z=xt[2])))
    report_chains('без MAP-старта', rows)


def hard_wall_prior(E=1.5, n=8, r_max=1.7):
    """x, y, z ~ U(-r_max, r_max) и отсечение точек вне шара вместо параметризации через Beta(3, 1)."""
    det = Detector(attenuation_length=8.0, rng=8501)
    g = np.random.default_rng(8502)
    rows = []
    for i in range(n):
        xt = random_vertex(g, 1.5)
        pe = det.response(xt, Source(rng=8600 + i).directions(E))
        with pm.Model():
            energy = pm.Uniform('E', 0, 10.0)
            x, y, z = [pm.Uniform(k, -r_max, r_max) for k in 'xyz']
            pos = pt.stack([x, y, z])
            pm.Potential('ball', pt.switch(pt.sum(pos**2) < r_max**2, 0.0, -np.inf))
            lam = rc.expected_photons(det, pos, energy, Y)
            h_max = int(2 * pe.max() / det.a + 40)
            pm.Potential('likelihood', rc.neyman_logp(pe, lam, det.a, h_max).sum())
            idata = pm.sample(500, tune=500, chains=2, cores=1, random_seed=i, progressbar=False, target_accept=0.9,
                              initvals=dict(E=E, x=0.1, y=0.1, z=0.1), compute_convergence_checks=False)
        rows.append(fit_summary(idata, dict(E=E, x=xt[0], y=xt[1], z=xt[2])))
    report_chains('приор с жёсткой стенкой', rows)


def larger_fiducial_volume(E=1.5, n=16):
    """r_max = 0.95 R вместо 0.85 R, вершины до 1.8 м."""
    det = Detector(attenuation_length=8.0, rng=8701)
    g = np.random.default_rng(8702)
    for r_max_rel, name in ((0.95, 'r_max = 0.95 R'), (0.85, 'r_max = 0.85 R')):
        rows = []
        for i in range(n):
            xt = random_vertex(g, 0.8 * r_max_rel * det.R)
            pe = det.response(xt, Source(rng=8800 + i).directions(E))
            model = rc.reconstruction_model(det, Y, pe, r_max=r_max_rel * det.R)
            init = rc.start_point(det, Y, pe, model)
            with model:
                idata = pm.sample(500, tune=500, chains=2, cores=1, random_seed=i, initvals=init, progressbar=False,
                                  target_accept=0.9, compute_convergence_checks=False)
            rows.append(fit_summary(idata, dict(E=E, x=xt[0], y=xt[1], z=xt[2])))
        report_chains(name, rows)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    charge_high_statistics()
    series(1.5, 1.5, 40, 200_000)
    series(3.0, 1.65, 40, 300_000)
    under_cathode()
    without_map_start()
    hard_wall_prior()
    larger_fiducial_volume()
