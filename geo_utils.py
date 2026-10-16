"""Гео-утилиты: расстояния (haversine), сферические метрики, координаты центра масс."""
from __future__ import annotations

import math
from typing import Iterable, Sequence

EARTH_R = 6_371_008.6  # средний радиус Земли, метры (WGS-84)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Расстояние по большому кругу между двумя точками, в метрах."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(a))


def distance_matrix_m(points: Sequence[tuple[float, float]]) -> "numpy.ndarray":
    """Матрица попарных расстояний (метры) для списка точек (lat, lon)."""
    import numpy as np

    arr = np.radians(np.asarray(points, dtype=float))
    lat, lon = arr[:, 0][:, None], arr[:, 1][:, None]
    dp = lat - lat.T
    dl = lon - lon.T
    a = np.sin(dp / 2) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dl / 2) ** 2
    d = 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return d


def path_length_m(coords: Sequence[tuple[float, float]]) -> float:
    """Суммарная длина ломаной, метры."""
    return sum(haversine_m(*coords[i], *coords[i + 1]) for i in range(len(coords) - 1))


def spherical_centroid(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """Центр масс на сфере (среднее по декартовым единичным векторам) — устойчиво к переходу через 180°."""
    import numpy as np

    arr = np.radians(np.asarray(points, dtype=float))
    x = np.cos(arr[:, 0]) * np.cos(arr[:, 1])
    y = np.cos(arr[:, 0]) * np.sin(arr[:, 1])
    z = np.sin(arr[:, 0])
    c = np.mean([x, y, z], axis=1)
    lon = math.atan2(c[1], c[0])
    hyp = math.hypot(c[0], c[1])
    lat = math.atan2(c[2], hyp)
    return math.degrees(lat), math.degrees(lon)


def parse_coords(text: str) -> tuple[float, float] | None:
    """Разбирает строку 'лат, лон' (через запятую/пробел/точку с запятой)."""
    if not text:
        return None
    cleaned = text.replace(";", ",").replace(" ", ",")
    parts = [p for p in cleaned.split(",") if p.strip()]
    if len(parts) != 2:
        return None
    try:
        lat, lon = float(parts[0]), float(parts[1])
    except ValueError:
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return lat, lon
