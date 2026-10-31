import numpy as np
from detector import Detector
from simulation import expected_fraction
from pymc_model import photon_model, multinomial_model, sample_Q

det = Detector(attenuation_length=8.0)
Y, E, draws = 10_000, 0.2, 300


def check(model, origin, attenuation):
    d = det if attenuation else Detector()
    s = sample_Q(model, draws, seed=3)
    Q_exp = d.a * Y * E * expected_fraction(d, origin).sum()
    assert abs(s['Q'].mean() - Q_exp) < 5 * np.sqrt(Q_exp * (1 + d.a) / draws)
    assert abs(s['n_gamma'].mean() - Y * E) < 5 * np.sqrt(Y * E / draws)


def test_photon_model():
    for att in (False, True):
        check(photon_model(det, Y, E, (0, 0, 1.0), attenuation=att), (0, 0, 1.0), att)


def test_multinomial_model():
    for att in (False, True):
        check(multinomial_model(det, Y, E, (0, 0, 1.0), attenuation=att), (0, 0, 1.0), att)
