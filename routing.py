"""Построение оптимальных маршрутов (TSP) по местам и по кластерам.

Подход двухуровневый:
  1. Межкластерный маршрут — порядок посещения кластеров решается как TSP
     на центроидах кластеров (эвристика nearest-neighbor + 2-opt).
  2. Внутрикластерный маршрут — обход мест внутри каждого кластера,
     начиная с «точки входа» (ближайшей к предыдущему кластеру), тоже TSP-эвристикой.

Для малых чисел точек (<= ~9) применяется точное решение перебором с ветвями
и ограничениями (branch & bound), иначе — жадная эвристика + 2-opt.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

from clustering import Cluster
from geo_utils import haversine_m, path_length_m
from places import Place


def dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return haversine_m(a[0], a[1], b[0], b[1])


# ---------------------------------------------------------------- базовые эвристики

def nearest_neighbor(coords: list[tuple[float, float]], start: int = 0) -> list[int]:
    """Жадный обход «ближайший сосед», возвращает порядок индексов."""
    n = len(coords)
    visited = [start]
    remaining = set(range(n)) - {start}
    while remaining:
        last = coords[visited[-1]]
        nxt = min(remaining, key=lambda j: dist(last, coords[j]))
        visited.append(nxt)
        remaining.remove(nxt)
    return visited


def two_opt(order: list[int], coords: list[tuple[float, float]],
            closed: bool = False, max_iter: int = 2000) -> list[int]:
    """Локальное улучшение 2-opt: убираем «самопересечения» маршрута."""
    def length(o):
        seg = list(o) + ([o[0]] if closed else [])
        return sum(dist(coords[seg[i]], coords[seg[i + 1]]) for i in range(len(seg) - 1))

    improved = True
    it = 0
    best = length(order)
    while improved and it < max_iter:
        improved = False
        it += 1
        for i in range(1, len(order) - 1):
            for j in range(i + 1, len(order)):
                cand = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                l = length(cand)
                if l < best - 1e-6:
                    order, best = cand, l
                    improved = True
    return order


def solve_tsp_exact_or_heuristic(coords: list[tuple[float, float]],
                                 start: int = 0, closed: bool = False) -> list[int]:
    """Порядок обхода точек. Точное решение для малых n, иначе NN+2-opt."""
    n = len(coords)
    if n <= 1:
        return list(range(n))
    if n <= 9:
        # branch & bound по перестановкам фиксированного старта
        best_order, best_len = None, math.inf
        rest = [i for i in range(n) if i != start]

        def bb(path: list[int], acc: float):
            nonlocal best_order, best_len
            if acc >= best_len:          # отсечение
                return
            if len(path) == n:
                total = acc + (dist(coords[path[-1]], coords[path[0]]) if closed else 0)
                if total < best_len:
                    best_len, best_order = total, list(path)
                return
            for j in rest:
                if j in path:
                    continue
                path.append(j)
                bb(path, acc + dist(coords[path[-2]], coords[j]))
                path.pop()

        bb([start], 0.0)
        if best_order is not None:
            return best_order
    order = nearest_neighbor(coords, start)
    return two_opt(order, coords, closed=closed)


# ---------------------------------------------------------------- структура маршрута

@dataclass
class RouteStop:
    place: Place | None      # None => «центр кластера» как цель
    cluster_id: int | None
    lat: float
    lon: float
    kind: str                # 'place' | 'cluster-center' | 'depot'
    name: str

    @property
    def coords(self):
        return self.lat, self.lon


@dataclass
class RouteResult:
    stops: list[RouteStop]
    total_km: float
    legs_by_cluster: dict[int, float] = field(default_factory=dict)
    inter_cluster_km: float = 0.0

    def summary_lines(self) -> list[str]:
        lines = [f"Маршрут: {len(self.stops)} остановок, общая длина ≈ {self.total_km:.2f} км"]
        for cid, km in sorted(self.legs_by_cluster.items()):
            lines.append(f"  внутри кластера {cid}: {km:.2f} км")
        lines.append(f"  межкластерные переезды: {self.inter_cluster_km:.2f} км")
        return lines


def _as_stop(p: Place, cid: int | None, kind: str = "place") -> RouteStop:
    return RouteStop(place=p, cluster_id=cid, lat=p.lat, lon=p.lon, kind=kind, name=p.label())


def route_within_cluster(cluster: Cluster, entry: tuple[float, float] | None = None) -> list[RouteStop]:
    """Оптимальный порядок мест внутри одного кластера.

    entry — точка, откуда въезжаем в кластер; первым посещаем ближайшую к ней точку.
    """
    pts = cluster.members
    if not pts:
        return []
    coords = [p.coords for p in pts]
    start = 0
    if entry is not None:
        start = min(range(len(coords)), key=lambda i: dist(entry, coords[i]))
    order = solve_tsp_exact_or_heuristic(coords, start=start, closed=False)
    return [_as_stop(pts[i], cluster.cid) for i in order]


def plan_route(clusters: list[Cluster], noise: list[Place] | None = None,
               depot: tuple[float, float] | None = None,
               mode: str = "clusters",
               include_noise: bool = True) -> RouteResult:
    """Строит сквозной маршрут.

    mode='clusters' — сначала оптимизация порядка кластеров (по центроидам),
                      затем обход мест внутри каждого кластера;
    mode='places'   — единый оптимальный маршрут по всем местам сразу.
    depot           — старт/финиш маршрута (например, дом или гостиница).
    """
    noise = noise or []
    result_stops: list[RouteStop] = []
    legs_by_cluster: dict[int, float] = {}
    inter_km = 0.0

    if not clusters:
        pts = list(noise)
        depot_stop = _depot_stop(depot) if depot else None
        seq: list[RouteStop] = []
        if pts:
            order = solve_tsp_exact_or_heuristic([p.coords for p in pts], start=0,
                                                 closed=bool(depot))
            seq = [_as_stop(pts[i], None) for i in order]
        if depot_stop:
            seq = [depot_stop] + seq + [depot_stop]
        total = path_length_m([s.coords for s in seq])
        return RouteResult(stops=seq, total_km=total / 1000)

    if mode == "places":
        pts: list[Place] = [p for c in clusters for p in c.members]
        if include_noise:
            pts += noise
        coords = [p.coords for p in pts]
        start = 0
        if depot:
            start = min(range(len(coords)), key=lambda i: dist(depot, coords[i]))
        order = solve_tsp_exact_or_heuristic(coords, start=start, closed=bool(depot))
        seq = [_as_stop(pts[i], next((c.cid for c in clusters if pts[i] in c.members), None))
               for i in order]
        if depot:
            seq = [_depot_stop(depot)] + seq + [_depot_stop(depot)]
        total = path_length_m([s.coords for s in seq])
        return RouteResult(stops=seq, total_km=total / 1000)

    # ---- mode == 'clusters': двухуровневая оптимизация
    centers = [c.center for c in clusters]
    start_idx = 0
    if depot:
        start_idx = min(range(len(centers)), key=lambda i: dist(depot, centers[i]))
    cluster_order = solve_tsp_exact_or_heuristic(centers, start=start_idx, closed=bool(depot))

    prev_point = depot
    for pos, ci in enumerate(cluster_order):
        cl = clusters[ci]
        # межкластерный участок: от prev_point до первой точки кластера
        inner_stops = route_within_cluster(cl, entry=prev_point)
        if prev_point and inner_stops:
            inter_km += dist(prev_point, inner_stops[0].coords) / 1000
        # длина внутрикластерных участков
        inner_len = 0.0
        for a, b in zip(inner_stops, inner_stops[1:]):
            inner_len += dist(a.coords, b.coords)
        legs_by_cluster[cl.cid] = legs_by_cluster.get(cl.cid, 0.0) + inner_len / 1000
        result_stops.extend(inner_stops)
        if inner_stops:
            prev_point = inner_stops[-1].coords

    # шумовые точки включаем отдельными «одиночными» остановками в конце,
    # вставляя каждую в наиболее дешёвую позицию сквозного маршрута
    if include_noise and noise:
        seq_coords = [depot] + [s.coords for s in result_stops] + ([depot] if depot else [])
        for p in noise:
            stop = _as_stop(p, None)
            if not seq_coords:  # кластеров нет — просто добавляем по порядку
                result_stops.append(stop)
                seq_coords.append(p.coords)
                continue
            best_i, best_delta = len(result_stops), math.inf
            for i in range(len(seq_coords) - 1):
                a, b = seq_coords[i], seq_coords[i + 1]
                delta = dist(a, p.coords) + dist(p.coords, b) - dist(a, b)
                if delta < best_delta:
                    best_delta, best_i = delta, i
            result_stops.insert(best_i, stop)
            seq_coords.insert(best_i + (1 if best_i < len(seq_coords) - 1 else 0), p.coords)
            inter_km += best_delta / 1000

    if depot:
        final_seq = [_depot_stop(depot)] + result_stops + [_depot_stop(depot)]
        if result_stops:
            inter_km += dist(depot, result_stops[0].coords) / 1000
            inter_km += dist(result_stops[-1].coords, depot) / 1000
    else:
        final_seq = result_stops

    total = path_length_m([s.coords for s in final_seq]) / 1000
    return RouteResult(stops=final_seq, total_km=total,
                       legs_by_cluster=legs_by_cluster, inter_cluster_km=inter_km)


def _depot_stop(depot: tuple[float, float]) -> RouteStop:
    return RouteStop(place=None, cluster_id=None, lat=depot[0], lon=depot[1],
                     kind="depot", name="Старт/Финиш")
