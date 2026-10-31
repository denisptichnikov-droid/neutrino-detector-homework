import logging
import warnings

import numpy as np
import pytensor.tensor as pt

from detector import Detector
from source import Source
from reconstruction import neyman_logp, reconstruct

logging.getLogger("pymc").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


def test_neyman_pmf():
    # нормировка, среднее a*lam и дисперсия a*lam*(1+a)
    lam, a = 3.0, 0.5
    m = np.arange(60)
    p = np.exp([neyman_logp(pt.constant(np.array([k])), pt.constant(np.array([lam])), a, 80).eval()
                for k in m])
    assert abs(p.sum() - 1) < 1e-9
    mean = (m * p).sum()
    assert abs(mean - a * lam) < 1e-9
    assert abs(((m - mean)**2 * p).sum() - a * lam * (1 + a)) < 1e-9


def test_reconstruct_event():
    det = Detector(attenuation_length=8.0, rng=5)
    x_true, E_true = np.array([0.4, 0.2, -0.6]), 2.0
    pe = det.response(x_true, Source(rng=5).directions(E_true))
    _, idata = reconstruct(det, 10_000, pe, draws=300, tune=300, chains=2, seed=5,
                           compute_convergence_checks=False)
    post = idata.posterior
    E = post['E'].values
    assert abs(E.mean() - E_true) < 3 * E.std()
    for k, v in zip('xyz', x_true):
        assert abs(post[k].values.mean() - v) < 3 * post[k].values.std()
