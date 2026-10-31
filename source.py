"""Источник сцинтилляционного света."""
import numpy as np

MeV = 1.0


class Source:
    """Изотропный источник света с выходом Y фотонов на 1 МэВ."""

    def __init__(self, Y: float = 10_000 / MeV, rng=None):
        self.Y = Y
        self.rng = np.random.default_rng(rng)

    def n_photons(self, E: float, fluctuate: bool = True) -> int:
        """Число фотонов для события с энергией E.

        Среднее линейно по энергии: <n> = Y*E. При fluctuate=True число
        разыгрывается по Пуассону (независимое излучение фотонов),
        иначе берётся округлённое среднее.
        """
        mean = self.Y * E
        if fluctuate:
            return int(self.rng.poisson(mean))
        return int(round(mean))

    def directions(self, E: float, fluctuate: bool = True) -> np.ndarray:
        """Начальные направления фотонов события с энергией E, массив (n, 3).

        Изотропия: phi ~ U(0, 2pi), cos(theta) ~ U(-1, 1) -- элемент
        телесного угла dOmega = dphi dcos(theta) равномерен.
        """
        n = self.n_photons(E, fluctuate)
        phi = self.rng.uniform(0, 2 * np.pi, n)
        cosTheta = self.rng.uniform(-1, 1, n)
        sinTheta = np.sqrt(1 - cosTheta**2)
        return np.stack([sinTheta * np.cos(phi),
                         sinTheta * np.sin(phi),
                         cosTheta], axis=1)
