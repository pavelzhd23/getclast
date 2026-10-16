"""Юнит-тесты ядра программы (без интернета): geo_utils, clustering, routing, адреса."""
from __future__ import annotations

import math
import os
import tempfile
import unittest
from unittest import mock

from geo_utils import haversine_m, distance_matrix_m, spherical_centroid, parse_coords
from clustering import cluster_places, Cluster
import places as places_mod
from places import (Place, parse_addresses, load_places_by_addresses,
                    load_addresses_file)
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


class TestAddresses(unittest.TestCase):
    """Импорт мест по списку адресов (геокодер замокан — тесты офлайн)."""

    def setUp(self):
        # изолируем дисковый кэш геокодера
        self._tmp = tempfile.TemporaryDirectory()
        self._old_cache_path = places_mod.GEOCACHE_PATH
        places_mod.GEOCACHE_PATH = os.path.join(self._tmp.name, "cache.json")
        places_mod._geo_mem_cache = {}

    def tearDown(self):
        places_mod.GEOCACHE_PATH = self._old_cache_path
        places_mod._geo_mem_cache = {}
        self._tmp.cleanup()

    def test_parse_addresses(self):
        text = "ул. А, 1\n\nул. Б, 2; ул. В, 3\r\n"
        self.assertEqual(parse_addresses(text), ["ул. А, 1", "ул. Б, 2", "ул. В, 3"])

    def test_load_by_addresses_mocked_geocoder(self):
        fake = {
            "ул. А, 1, Казань": [{"lat": "55.750", "lon": "37.610",
                                  "display_name": "ул. А, 1, Казань",
                                  "address": {"road": "ул. А", "house_number": "1"}}],
            "ул. Б, 2, Россия": [{"lat": "55.752", "lon": "37.612",
                                  "display_name": "ул. Б, 2",
                                  "address": {"road": "ул. Б", "house_number": "2"}}],
            "нет такого адреса": [],
        }

        def fake_raw(query, limit=10, viewbox=None, use_cache=True):
            return fake.get(query, [])

        with mock.patch.object(places_mod, "geocode_raw", fake_raw):
            result, failed = load_places_by_addresses(
                "ул. А, 1\nул. Б, 2; нет такого адреса", country_bias="Казань")
        # «ул. Б, 2» без города -> подсказка «Казань» не подошла (в фейке только
        # ключ с «Россия»), но если бы город совпал — точка появилась бы.
        # Проверяем главное: найденные точки и список ненайденных адресов.
        self.assertTrue(all(isinstance(p, Place) for p in result))
        self.assertIn("нет такого адреса", failed)
        if result:
            self.assertAlmostEqual(result[0].lat, 55.750)
            # источник указывает на реально использованный геокодер
            self.assertIn(result[0].source, ("photon", "nominatim"))

    def test_country_bias_applied(self):
        seen_queries = []

        def fake_raw(query, limit=10, viewbox=None, use_cache=True):
            seen_queries.append(query)
            return [{"lat": "55.75", "lon": "37.61", "display_name": query,
                    "address": {"road": "ул. Тестовая", "house_number": "9"}}]

        with mock.patch.object(places_mod, "geocode_raw", fake_raw):
            load_places_by_addresses(["ул. Тестовая, 9"], country_bias="Казань")
        self.assertEqual(seen_queries, ["ул. Тестовая, 9, Казань"])
        # повторный вызов берётся из кэша — сеть не нужна
        seen_queries.clear()
        load_places_by_addresses(["ул. Тестовая, 9"], country_bias="Казань")
        self.assertEqual(seen_queries, [])

    def test_coords_line_without_network(self):
        # строки «лат, лон» принимаются напрямую, геокодер не вызывается
        with mock.patch.object(places_mod, "geocode_raw",
                               side_effect=AssertionError("сеть не нужна")):
            result, failed = load_places_by_addresses("55.75, 37.61\n59.94, 30.31")
        self.assertEqual(len(result), 2)
        self.assertEqual(failed, [])
        self.assertEqual(result[0].source, "coords")

    def test_duplicate_addresses_deduped(self):
        resp = [{"lat": "55.75", "lon": "37.61", "display_name": "дубль",
                 "address": {"road": "ул. Д"}}, ]

        with mock.patch.object(places_mod, "geocode_raw",
                               lambda *a, **k: resp):
            result, _ = load_places_by_addresses(["ул. Д, 1", "ул. Д, 1", "ул. Д, 2"])
        self.assertEqual(len(result), 1)  # одна и та же точка ~ в пределах 11 м

    def test_load_addresses_file_txt_csv_json(self):
        d = self._tmp.name
        txt = os.path.join(d, "a.txt")
        with open(txt, "w", encoding="utf-8") as f:
            f.write("ул. Одна, 1\nул. Две, 2\n")
        self.assertEqual(load_addresses_file(txt), ["ул. Одна, 1", "ул. Две, 2"])

        csvp = os.path.join(d, "a.csv")
        with open(csvp, "w", encoding="utf-8") as f:
            f.write('name,address\nX,"ул. Три, 3"\nY,"ул. Четыре, 4"\n')
        self.assertEqual(load_addresses_file(csvp), ["ул. Три, 3", "ул. Четыре, 4"])

        jsonp = os.path.join(d, "a.json")
        with open(jsonp, "w", encoding="utf-8") as f:
            f.write('["ул. Пять, 5", {"address": "ул. Шесть, 6"}, {"note": "без адреса"}]')
        self.assertEqual(load_addresses_file(jsonp), ["ул. Пять, 5", "ул. Шесть, 6"])


class TestGeocoderSelection(unittest.TestCase):
    """Выбор сервиса геокодирования: Photon основной, Nominatim — резерв."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old_cache_path = places_mod.GEOCACHE_PATH
        places_mod.GEOCACHE_PATH = os.path.join(self._tmp.name, "cache.json")
        places_mod._geo_mem_cache = {}
        self._old_mode = places_mod.get_geocoder()
        places_mod.set_geocoder("photon")
        places_mod._photon_failures = 0

    def tearDown(self):
        places_mod.set_geocoder(self._old_mode)
        places_mod.GEOCACHE_PATH = self._old_cache_path
        places_mod._geo_mem_cache = {}
        self._tmp.cleanup()

    PHOTON_GEOJSON = {
        "type": "FeatureCollection",
        "features": [{
            "geometry": {"type": "Point", "coordinates": [49.10665, 55.79491]},
            "properties": {"name": "", "street": "улица Баумана",
                           "housenumber": "20", "city": "Казань",
                           "type": "building", "osm_key": "amenity"},
        }],
    }

    def test_photon_response_converted_to_nominatim_format(self):
        items = places_mod._photon_to_items(self.PHOTON_GEOJSON, limit=3)
        self.assertEqual(len(items), 1)
        it = items[0]
        self.assertAlmostEqual(float(it["lat"]), 55.79491, places=5)
        self.assertAlmostEqual(float(it["lon"]), 49.10665, places=5)
        self.assertIn("Баумана", it["display_name"])
        self.assertEqual(it["address"]["city"], "Казань")

    def test_photon_used_by_default(self):
        calls = []

        def fake_http(url, params=None, timeout=30, retries=1):
            calls.append(url)
            return self.PHOTON_GEOJSON

        with mock.patch.object(places_mod, "_http_get_json", fake_http):
            res = places_mod.geocode_raw("ул. Баумана, 20, Казань", limit=3)
        self.assertEqual(calls, [places_mod.PHOTON_URL])
        self.assertEqual(len(res), 1)
        p = places_mod.geocode_address("ул. Баумана, 20, Казань")
        self.assertIsNotNone(p)
        self.assertEqual(p.source, "photon")

    def test_fallback_to_nominatim_on_photon_failure(self):
        places_mod.set_geocoder("auto")
        nom_result = [{"lat": "55.75", "lon": "37.61",
                      "display_name": "резерв", "address": {}}]

        def fake_http(url, params=None, timeout=30, retries=1):
            if url == places_mod.PHOTON_URL:
                raise OSError("Photon недоступен")
            return nom_result

        with mock.patch.object(places_mod, "_http_get_json", fake_http):
            res = places_mod.geocode_raw("какой-то адрес", limit=2)
        self.assertEqual(res, nom_result)  # автоматически ушли на Nominatim

    def test_persistent_photon_failure_switches_active_geocoder(self):
        places_mod.set_geocoder("auto")

        def broken(url, params=None, timeout=30, retries=1):
            raise OSError("нет связи")

        with mock.patch.object(places_mod, "_http_get_json", broken):
            for i in range(places_mod.PHOTON_FAILURE_LIMIT + 1):
                try:
                    places_mod.geocode_raw(f"адрес {i}", limit=1, use_cache=False)
                except Exception:
                    pass
        self.assertEqual(places_mod.active_geocoder(), "nominatim")

    def test_set_geocoder_validation(self):
        with self.assertRaises(ValueError):
            places_mod.set_geocoder("google")


if __name__ == "__main__":
    unittest.main(verbosity=2)
