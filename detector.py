"""Геометрия сферического жидкосцинтилляционного детектора."""
import numpy as np

m, cm = 1.0, 1e-2

vector3 = np.ndarray  # shape (N, 3)


def fibonacci_sphere(N: int) -> vector3:
    """N почти равномерно распределённых единичных векторов (спираль Фибоначчи).

    Точки ставятся на равные по площади пояса (равномерно по cos(theta)),
    а азимут сдвигается на «золотой угол» -- так соседи не выстраиваются в линии.
    """
    i = np.arange(N) + 0.5
    cosTheta = 1 - 2 * i / N
    sinTheta = np.sqrt(1 - cosTheta**2)
    phi = np.pi * (3 - np.sqrt(5)) * i
    return np.stack([sinTheta * np.cos(phi),
                     sinTheta * np.sin(phi),
                     cosTheta], axis=1)


class Detector:
    """Сфера радиуса R, на поверхности которой стоят N ФЭУ.

    Фотокатод ФЭУ -- плоский диск радиуса r, вписанный в сферу (край на
    поверхности), с осью `pmt_directions[k]`; `pmt_positions[k]` -- точка
    сферы на этой оси.
    """

    def __init__(self, R: float = 2 * m, N: int = 40,
                 r: float = 10 * cm, a: float = 0.5,
                 attenuation_length: float = np.inf, rng=None,
                 search: str = 'brute'):
        self.R = R
        self.N = N
        self.r = r
        self.a = a
        # длина затухания света в сцинтилляторе; inf -- без затухания
        self.attenuation_length = attenuation_length
        self.rng = np.random.default_rng(rng)
        # массив (N, 3): строка k -- координаты центра k-го ФЭУ
        self.pmt_directions = fibonacci_sphere(N)
        self.pmt_positions = R * self.pmt_directions
        # алгоритм поиска ФЭУ: 'brute' -- перебор всех ФЭУ,
        # 'grid' -- сетка по (cos theta, phi), 'kdtree' -- дерево scipy
        self.search = search
        self._grid = None
        self._tree = None

    def pmt_index(self, points: np.ndarray) -> np.ndarray:
        """Номер ФЭУ, которому принадлежит точка на поверхности, или -1.

        Выбор алгоритма -- атрибут `search`; результат у всех одинаковый.
        """
        points = np.atleast_2d(points)
        if self.search == 'grid':
            return self.pmt_index_grid(points)
        if self.search == 'kdtree':
            return self.pmt_index_kdtree(points)
        return self.pmt_index_brute(points)

    def _on_cap(self, points: np.ndarray, k: np.ndarray) -> np.ndarray:
        """Лежит ли точка points[i] на катоде k[i] (k = -1 -- нет)."""
        n = self.pmt_directions[np.maximum(k, 0)]
        cos = np.sum(points * n, axis=-1)
        rho2 = np.sum(points**2, axis=-1) - cos**2
        return (k >= 0) & (cos > 0) & (rho2 <= self.r**2)

    def build_grid(self, n_cos: int = None, n_phi: int = None):
        """Сетка ячеек по (cos theta, phi) со списками ФЭУ-кандидатов.

        Ячейки равны по площади (равный шаг по cos theta). Для каждой ячейки
        запоминаются ФЭУ, чей катод может её задевать: угловое расстояние от
        центра ячейки до оси ФЭУ не больше alpha + половины углового
        размера ячейки. Списки дополняются -1 до одинаковой длины K, так что
        кандидаты хранятся в массиве (n_cells, K) и выбираются индексацией.
        """
        if n_cos is None:
            # порядка одного ФЭУ на ячейку
            n_cos = max(4, int(np.sqrt(self.N / 2)))
        if n_phi is None:
            n_phi = 2 * n_cos
        cos_edges = np.linspace(-1, 1, n_cos + 1)
        phi_edges = np.linspace(-np.pi, np.pi, n_phi + 1)

        # углы, ограничивающие ячейку: точки по контуру ячейки
        s = np.linspace(0, 1, 5)
        c_lo, c_hi = cos_edges[:-1, None], cos_edges[1:, None]
        p_lo, p_hi = phi_edges[:-1, None], phi_edges[1:, None]
        cos_c = 0.5 * (c_lo + c_hi)[:, 0]
        phi_c = 0.5 * (p_lo + p_hi)[:, 0]

        def unit(cos, phi):
            cos, phi = np.broadcast_arrays(cos, phi)
            sin = np.sqrt(np.clip(1 - cos**2, 0, None))
            return np.stack([sin * np.cos(phi), sin * np.sin(phi), cos], axis=-1)

        CC, PC = np.meshgrid(cos_c, phi_c, indexing='ij')
        centers = unit(CC, PC)                                   # (n_cos, n_phi, 3)
        # контур ячейки: стороны cos = const и phi = const
        cos_b = np.concatenate([np.broadcast_to(c_lo, (n_cos, 5)),
                                np.broadcast_to(c_hi, (n_cos, 5)),
                                c_lo + s * (c_hi - c_lo),
                                c_lo + s * (c_hi - c_lo)], axis=1)  # (n_cos, 20)
        phi_s = np.concatenate([p_lo + s * (p_hi - p_lo),
                                p_lo + s * (p_hi - p_lo),
                                np.broadcast_to(p_lo, (n_phi, 5)),
                                np.broadcast_to(p_hi, (n_phi, 5))], axis=1)  # (n_phi, 20)
        border = unit(cos_b[:, None, :], phi_s[None, :, :])     # (n_cos, n_phi, 20, 3)
        cos_half = np.min(np.sum(border * centers[:, :, None], axis=-1), axis=-1)
        half = np.arccos(np.clip(cos_half, -1, 1))               # полуразмер ячейки

        alpha = np.arcsin(self.r / self.R)
        ang = np.arccos(np.clip(centers.reshape(-1, 3) @ self.pmt_directions.T, -1, 1))
        cand = ang <= (alpha + half.reshape(-1, 1)) * 1.01      # (n_cells, N)
        K = max(1, cand.sum(axis=1).max())
        table = np.full((cand.shape[0], K), -1)
        cells, pmts = np.nonzero(cand)
        pos = np.arange(len(cells)) - np.searchsorted(cells, cells)  # номер в строке
        table[cells, pos] = pmts
        self._grid = dict(n_cos=n_cos, n_phi=n_phi, table=table)
        return self._grid

    def pmt_index_grid(self, points: np.ndarray) -> np.ndarray:
        """Поиск через сетку: проверяются только K кандидатов ячейки, а не все N."""
        if self._grid is None:
            self.build_grid()
        g = self._grid
        rho = np.linalg.norm(points, axis=1)
        cos = points[:, 2] / rho
        phi = np.arctan2(points[:, 1], points[:, 0])
        i = np.clip(((cos + 1) / 2 * g['n_cos']).astype(int), 0, g['n_cos'] - 1)
        j = np.clip(((phi + np.pi) / (2 * np.pi) * g['n_phi']).astype(int), 0, g['n_phi'] - 1)
        cand = g['table'][i * g['n_phi'] + j]                   # (M, K)
        hit = self._on_cap(points[:, None, :], cand)            # (M, K)
        return np.where(hit.any(axis=1), cand[np.arange(len(points)), hit.argmax(axis=1)], -1)

    def pmt_index_kdtree(self, points: np.ndarray) -> np.ndarray:
        """Поиск через k-d дерево: ближайший ФЭУ за O(log N), затем проверка катода.

        Катоды не перекрываются, поэтому если точка лежит на катоде, то центр
        этого ФЭУ -- ближайший к ней (угловое расстояние < alpha, а до любого
        другого ФЭУ > alpha).
        """
        if self._tree is None:
            from scipy.spatial import cKDTree
            self._tree = cKDTree(self.pmt_directions)
        rho = np.linalg.norm(points, axis=1, keepdims=True)
        _, k = self._tree.query(points / rho)
        return np.where(self._on_cap(points, k), k, -1)

    def pmt_index_brute(self, points: np.ndarray) -> np.ndarray:
        """Номер ФЭУ, которому принадлежит точка на поверхности, или -1.

        points: массив (3,) или (M, 3) точек на сфере радиуса R.
        Точка принадлежит ФЭУ k, если её расстояние до оси катода
        (поперечная компонента относительно n_k) не больше r и она лежит
        в той же полусфере, что и катод. Такая область -- проекция диска
        на сферу, т.е. точно то, что «видит» плоский фотокатод.
        """
        points = np.atleast_2d(points)                      # (M, 3)
        cos = points @ self.pmt_directions.T                # (M, N): p·n_k
        p2 = np.sum(points**2, axis=1, keepdims=True)       # (M, 1)
        rho2 = p2 - cos**2                                  # |p_T|^2
        hit = (rho2 <= self.r**2) & (cos > 0)               # (M, N)
        # при разумной расстановке катоды не перекрываются, так что
        # в строке не больше одного True; argmax берёт его номер
        index = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
        return index

    def intersect_surface(self, origins: np.ndarray,
                          directions: np.ndarray) -> np.ndarray:
        """Точки, в которых лучи из `origins` вдоль `directions` выходят на сферу.

        origins: (3,) или (M, 3) -- точки внутри детектора;
        directions: (M, 3) -- единичные векторы.
        Решаем |o + t n|^2 = R^2:  t^2 + 2 (o·n) t + |o|^2 - R^2 = 0.
        Изнутри сферы |o| < R, свободный член < 0, поэтому корни разных
        знаков и нужный (вперёд по лучу) -- положительный.
        """
        origins = np.broadcast_to(origins, directions.shape)
        t = self.path_length(origins, directions)
        return origins + t[:, None] * directions

    def path_length(self, origins: np.ndarray,
                    directions: np.ndarray) -> np.ndarray:
        """Расстояние от `origins` до стенки вдоль `directions`, массив (M,)."""
        origins = np.broadcast_to(origins, directions.shape)
        b = np.sum(origins * directions, axis=1)
        c = np.sum(origins**2, axis=1) - self.R**2
        return -b + np.sqrt(b**2 - c)

    def survives(self, path: np.ndarray) -> np.ndarray:
        """Розыгрыш затухания: True, если фотон прошёл путь `path` без поглощения.

        Вероятность пройти путь l: exp(-l / L), L -- длина затухания.
        """
        if np.isinf(self.attenuation_length):
            return np.ones(len(path), dtype=bool)
        p = np.exp(-path / self.attenuation_length)
        return self.rng.uniform(size=len(path)) < p

    def trace(self, origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
        """Номер ФЭУ, в который попадает каждый фотон, или -1.

        Без рассеяния фотон летит по прямой до стенки, где поглощается;
        по пути он может быть поглощён в сцинтилляторе (см. `survives`).
        Катод -- плоский диск радиуса r, вписанный в сферу (его край лежит
        на поверхности). Диск и сферический сегмент за ним ограничивают
        замкнутый объём, поэтому луч из детектора пересекает диск тогда и
        только тогда, когда выходит на сферу внутри этого сегмента -- это
        ровно условие `pmt_index`. Путь до катода считаем равным пути до
        сферы: разница не больше R - sqrt(R^2 - r^2) ≈ 2.5 мм << L.
        """
        origins = np.broadcast_to(origins, directions.shape)
        t = self.path_length(origins, directions)
        index = self.pmt_index(origins + t[:, None] * directions)
        index[~self.survives(t)] = -1
        return index

    def photoelectrons(self, n_photons: np.ndarray) -> np.ndarray:
        """Случайное число фотоэлектронов для каждого ФЭУ.

        n_photons: массив (N,) -- число фотонов, попавших на каждый катод.
        Каждый фотон даёт Poisson(a) фотоэлектронов; сумма независимых
        пуассоновских величин -- снова Пуассон, со средним a * n_photons.
        """
        return self.rng.poisson(self.a * np.asarray(n_photons))

    def response(self, origins: np.ndarray,
                 directions: np.ndarray) -> np.ndarray:
        """Число фотоэлектронов в каждом из N каналов для набора фотонов."""
        idx = self.trace(origins, directions)
        n_photons = np.bincount(idx[idx >= 0], minlength=self.N)
        return self.photoelectrons(n_photons)

    def is_on_pmt(self, points: np.ndarray) -> np.ndarray:
        """True, если точка принадлежит какому-либо ФЭУ."""
        return self.pmt_index(points) >= 0

    def coverage(self) -> float:
        """Доля поверхности сферы, покрытая катодами (аналитически).

        Площадь «шапки» под катодом: 2 pi R^2 (1 - cos alpha), sin alpha = r/R.
        """
        cos_alpha = np.sqrt(1 - (self.r / self.R)**2)
        return self.N * (1 - cos_alpha) / 2

    def min_pmt_distance(self) -> float:
        """Минимальное расстояние между центрами соседних ФЭУ."""
        d = np.linalg.norm(self.pmt_positions[:, None] - self.pmt_positions[None],
                           axis=2)
        np.fill_diagonal(d, np.inf)
        return d.min()
