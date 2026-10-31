"""Эталонная реализация на чистом Python: фотоны по одному, списки и модуль math.

Нужна только для сравнения скорости с векторной версией (detector.py, source.py)
и для проверки, что обе версии дают одинаковую статистику.
"""
import math
import random


def fibonacci_sphere(N):
    golden = math.pi * (3 - math.sqrt(5))
    points = []
    for i in range(N):
        cosTheta = 1 - 2 * (i + 0.5) / N
        sinTheta = math.sqrt(1 - cosTheta**2)
        phi = golden * (i + 0.5)
        points.append((sinTheta * math.cos(phi), sinTheta * math.sin(phi), cosTheta))
    return points


class PyDetector:
    def __init__(self, R=2.0, N=40, r=0.10, a=0.5, attenuation_length=math.inf):
        self.R, self.N, self.r, self.a = R, N, r, a
        self.attenuation_length = attenuation_length
        self.pmt_directions = fibonacci_sphere(N)

    def path_length(self, o, n):
        b = sum(oi * ni for oi, ni in zip(o, n))
        c = sum(oi * oi for oi in o) - self.R**2
        return -b + math.sqrt(b * b - c)

    def pmt_index(self, p):
        p2 = sum(x * x for x in p)
        for k, nk in enumerate(self.pmt_directions):
            cos = sum(pi * ni for pi, ni in zip(p, nk))
            if cos > 0 and p2 - cos * cos <= self.r**2:
                return k
        return -1

    def trace(self, o, n):
        t = self.path_length(o, n)
        if random.random() >= math.exp(-t / self.attenuation_length):
            return -1
        p = tuple(oi + t * ni for oi, ni in zip(o, n))
        return self.pmt_index(p)

    def photoelectrons(self, n_photons):
        return [poisson(self.a * k) for k in n_photons]

    def response(self, o, directions):
        hits = [0] * self.N
        for n in directions:
            k = self.trace(o, n)
            if k >= 0:
                hits[k] += 1
        return self.photoelectrons(hits)


class PySource:
    def __init__(self, Y=10_000):
        self.Y = Y

    def directions(self, E):
        result = []
        for _ in range(poisson(self.Y * E)):
            phi = random.uniform(0, 2 * math.pi)
            cosTheta = random.uniform(-1, 1)
            sinTheta = math.sqrt(1 - cosTheta**2)
            result.append((sinTheta * math.cos(phi), sinTheta * math.sin(phi), cosTheta))
        return result


def poisson(mean):
    """Пуассон: алгоритм Кнута для малых средних, нормальное приближение для больших."""
    if mean <= 0:
        return 0
    if mean > 50:
        return max(0, round(random.gauss(mean, math.sqrt(mean))))
    L, k, p = math.exp(-mean), 0, 1.0
    while True:
        p *= random.random()
        if p <= L:
            return k
        k += 1
