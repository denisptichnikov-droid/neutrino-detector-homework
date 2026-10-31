"""Моделирование серий событий."""
import numpy as np
from detector import Detector
from source import Source


def simulate_charge(det: Detector, src: Source, E: float,
                    origin=(0, 0, 0), n_events: int = 1000) -> np.ndarray:
    """Заряд (ф.э.) в каждом канале для n_events событий, массив (n_events, N)."""
    origin = np.asarray(origin, dtype=float)
    return np.array([det.response(origin, src.directions(E))
                     for _ in range(n_events)])


def expected_fraction(det: Detector, origin=(0, 0, 0)) -> np.ndarray:
    """Доля фотонов из точки origin, попадающая на каждый ФЭУ (приближение малого диска).

    Телесный угол диска площадью pi r^2 на расстоянии d, видимого под углом psi
    к его оси: Omega ≈ pi r^2 cos(psi) / d^2. Центр диска -- на плоскости
    вписанного катода, на расстоянии sqrt(R^2 - r^2) от центра сферы.
    """
    origin = np.asarray(origin, dtype=float)
    centers = np.sqrt(det.R**2 - det.r**2) * det.pmt_directions
    v = centers - origin
    d = np.linalg.norm(v, axis=1)
    cos_psi = np.sum(v * det.pmt_directions, axis=1) / d
    return det.r**2 * cos_psi / (4 * d**2)
