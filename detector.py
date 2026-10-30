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

    Фотокатод ФЭУ -- плоский диск радиуса r, касающийся сферы в точке
    `pmt_positions[k]`, с нормалью `pmt_directions[k]` (наружу).
    """

    def __init__(self, R: float = 2 * m, N: int = 40,
                 r: float = 10 * cm, a: float = 0.5):
        self.R = R
        self.N = N
        self.r = r
        self.a = a
        # массив (N, 3): строка k -- координаты центра k-го ФЭУ
        self.pmt_directions = fibonacci_sphere(N)
        self.pmt_positions = R * self.pmt_directions

    def pmt_index(self, points: np.ndarray) -> np.ndarray:
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
