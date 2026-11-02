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

    def _isotropic(self, n: int) -> np.ndarray:
        """n изотропных единичных векторов, массив (n, 3)."""
        phi = self.rng.uniform(0, 2 * np.pi, n)
        cosTheta = self.rng.uniform(-1, 1, n)
        sinTheta = np.sqrt(1 - cosTheta**2)
        return np.stack([sinTheta * np.cos(phi),
                         sinTheta * np.sin(phi),
                         cosTheta], axis=1)

    def photons(self, E: float, fluctuate: bool = True):
        """Генератор: направления фотонов события по одному, массивы (3,).

        Число фотонов разыгрывается один раз в начале события, дальше
        направления выдаются лениво -- в памяти держится только текущий фотон.
        """
        n = self.n_photons(E, fluctuate)
        for _ in range(n):
            yield self._isotropic(1)[0]

    def photon_batches(self, E: float, batch_size: int = 100,
                       fluctuate: bool = True):
        """Генератор: направления фотонов события пачками, массивы (k, 3).

        Все пачки, кроме, возможно, последней, содержат batch_size фотонов;
        последняя -- остаток, так что в сумме выдаётся ровно n_photons(E).
        Внутри пачки вычисления векторные (numpy), а память ограничена
        размером пачки.
        """
        n = self.n_photons(E, fluctuate)
        for start in range(0, n, batch_size):
            yield self._isotropic(min(batch_size, n - start))

    def directions(self, E: float, fluctuate: bool = True) -> np.ndarray:
        """Начальные направления фотонов события с энергией E, массив (n, 3).

        Изотропия: phi ~ U(0, 2pi), cos(theta) ~ U(-1, 1) -- элемент
        телесного угла dOmega = dphi dcos(theta) равномерен.
        """
        return self._isotropic(self.n_photons(E, fluctuate))
