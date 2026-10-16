"""Юнит-тесты ядра программы (без интернета): geo_utils, clustering, routing."""
from __future__ import annotations

import math
import unittest

from geo_utils import haversine_m, distance_matrix_m, spherical_centroid, parse_coords
from clustering import cluster_places, Cluster
from places import Place
from routing import plan_route, solve_tsp_exact_or_heuristic, route_within_cluster


class TestGeo(unittest.TestCase):
    def test_haversine_known_distance(self):
        # Москва (55.7558,37.6173) -> Санкт-Петербург (59.9343,30.3351) ≈ 634 км
        d = haversine_m(55.7558, 37.6173, 59.9343, 30.3351)
        self.assertAlmostEqual(d / 1000, 634, delta=10)

    def test_haversine_zero(self):
        self.assertEqual(haversine_m(55.0, 37.0, 55.0, 37.0), 0.0)

    def test_distance_matrix_symmetry(self):
        pts = [(55.75, 37.62), (55.76, 37.63), (59.93, 30.33)]
        D = distance_matrix_m(pts)
        self.assertEqual(D.shape, (3, 3))
        self.assertTrue(math.isclose(D[0, 2], D[2, 0]))
        self.assertEqual(D[0, 0], 0)

    def test_centroid(self):
        c = spherical_centroid([(55.0, 37.0), (55.0, 37.001)])
        self.assertAlmostEqual(c[0], 55.0, places=5)
        self.assertAlmostEqual(c[1], 37.0005, places=4)

    def test_parse_coords(self):
        self.assertEqual(parse_coords("55.75, 37.62"), (55.75, 37.62))
        self.assertEqual(parse_coords("55.75;37.62"), (55.75, 37.62))
        self.assertIsNone(parse_coords("abc"))
        self.assertIsNone(parse_coords("100, 37"))


class TestClustering(unittest.TestCase):
    @staticmethod
    def _mk():
        # два плотных скопления ~500 м друг от друга + одна далёкая точка
        a = [Place(f"A{i}", 55.750 + i * 0.0005, 37.610) for i in range(4)]
        b = [Place(f"B{i}", 55.760 + i * 0.0005, 37.610) for i in range(4)]
        far = Place("F", 55.900, 37.900)
        return a + b + [far]

    def test_dbscan_two_clusters_one_noise(self):
        clusters, noise = cluster_places(self._mk(), eps_m=150, min_samples=2)
        self.assertEqual(len(clusters), 2)
        self.assertEqual([c.size for c in clusters], [4, 4])
        self.assertEqual(len(noise), 1)

    def test_hdbscan_fallback_runs(self):
        clusters, noise = cluster_places(self._mk(), method="hdbscan", min_samples=2)
        self.assertGreaterEqual(len(clusters), 1)

    def test_cluster_center_and_radius(self):
        cl = Cluster(cid=1, members=[Place("x", 55.75, 37.61), Place("y", 55.752, 37.61)])
        lat, lon = cl.center
        self.assertAlmostEqual(lat, 55.751, places=3)
        self.assertGreater(cl.radius_m(), 0)


class TestRouting(unittest.TestCase):
    def test_exact_tsp_small(self):
        # квадрат 4 точки: оптимальный открытый путь из точки 0 = 2 ребра стороны
        sq = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
        order = solve_tsp_exact_or_heuristic(sq, start=0)
        self.assertEqual(sorted(order), [0, 1, 2, 3])

    def test_plan_route_visits_all(self):
        pts = [Place(f"P{i}", 55.750 + (i % 3) * 0.001, 37.610 + (i // 3) * 0.001)
               for i in range(9)]
        clusters, noise = cluster_places(pts, eps_m=200, min_samples=2)
        r = plan_route(clusters, noise=noise, mode="clusters")
        names = {s.name for s in r.stops}
        self.assertEqual(names, {p.label() for p in pts})
        self.assertGreater(r.total_km, 0)

    def test_depot_closed_loop(self):
        pts = [Place("A", 55.75, 37.61), Place("B", 55.76, 37.62),
               Place("C", 55.77, 37.63), Place("D", 55.755, 37.615)]
        clusters, noise = cluster_places(pts, eps_m=5000, min_samples=2)
        depot = (55.74, 37.60)
        r = plan_route(clusters, noise=noise, depot=depot, mode="clusters")
        self.assertEqual(r.stops[0].kind, "depot")
        self.assertEqual(r.stops[-1].kind, "depot")

    def test_mode_places_vs_clusters_length(self):
        # маршрут «по кластерам» не должен быть катастрофически длиннее прямого
        pts = ([Place(f"L{i}", 55.750 + i * 0.0004, 37.610) for i in range(5)] +
               [Place(f"R{i}", 55.750 + i * 0.0004, 37.660) for i in range(5)])
        clusters, noise = cluster_places(pts, eps_m=150, min_samples=2)
        rc = plan_route(clusters, noise=noise, mode="clusters")
        rp = plan_route(clusters, noise=noise, mode="places")
        self.assertLess(rc.total_km, rp.total_km * 1.5)

    def test_empty_input(self):
        r = plan_route([], noise=[], depot=None, mode="clusters")
        self.assertEqual(r.stops, [])
        self.assertEqual(r.total_km, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
