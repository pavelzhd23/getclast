"""CLI: поиск мест → кластеризация по плотности → оптимальный маршрут → HTML-карта.

Примеры:
  python cli.py --demo --out map.html
  python cli.py --category "Кафе и рестораны" --city "Казань" --radius-km 4 \
                --eps 500 --min-samples 3 --mode clusters --depot "55.796,49.106" --out map.html
  python cli.py --query "музеи центра Алматы" --limit 40 --eps 800 --out map.html
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys

from clustering import cluster_places, silhouette_check
from geo_utils import parse_coords
from places import CATEGORY_TAGS, Place, find_places, geocode
from routing import plan_route
from visualization import render_map


def load_places_file(path: str) -> list[Place]:
    with open(path, encoding="utf-8-sig") as f:
        text = f.read()
    if path.endswith(".json"):
        rows = json.loads(text)
    else:
        import csv
        rows = list(csv.DictReader(io.StringIO(text)))
    out = []
    for r in rows:
        try:
            out.append(Place(str(r.get("name", "")), float(r["lat"]), float(r["lon"]),
                             address=str(r.get("address", "")),
                             category=str(r.get("category", ""))))
        except (KeyError, ValueError):
            continue
    return out


DEMO_PLACES = [
    Place("Эрмитаж", 59.9398, 30.3146, category="Музеи"),
    Place("Русский музей", 59.9385, 30.3370, category="Музеи"),
    Place("Спас на Крови", 59.9407, 30.3320, category="Достопримечательности"),
    Place("Казанский собор", 59.9358, 30.3264, category="Достопримечательности"),
    Place("Исаакиевский собор", 59.9304, 30.1939, category="Достопримечательности"),
    Place("Петропавловская крепость", 59.9504, 30.3217, category="Достопримечательности"),
    Place("Стрелка ВО", 59.9466, 30.2470, category="Достопримечательности"),
    Place("Крейсер Аврора", 59.9483, 30.2407, category="Достопримечательности"),
    Place("Новая Голландия", 59.9342, 30.2899, category="Парки"),
    Place("Марсово поле", 59.9411, 30.3378, category="Парки"),
    Place("Летний сад", 59.9436, 30.3356, category="Парки"),
    Place("Гостиный двор", 59.9375, 30.3386, category="Магазины"),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--demo", action="store_true", help="использовать демо-набор мест (СПб)")
    src.add_argument("--places-file", help="CSV/JSON файл со столбцами name,lat,lon[,category,address]")
    ap.add_argument("--query", help="текстовый запрос к Nominatim")
    ap.add_argument("--category", choices=list(CATEGORY_TAGS), help="категория OSM через Overpass")
    ap.add_argument("--city", default="Москва", help="город для границы поиска")
    ap.add_argument("--radius-km", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--method", default="dbscan", choices=["dbscan", "hdbscan"])
    ap.add_argument("--eps", type=float, default=500.0, help="радиус окрестности, метры")
    ap.add_argument("--min-samples", type=int, default=3)
    ap.add_argument("--mode", default="clusters", choices=["clusters", "places"],
                    help="уровень оптимизации маршрута")
    ap.add_argument("--depot", help="старт/финиш 'лат,лон'")
    ap.add_argument("--no-noise", action="store_true", help="не включать места вне кластеров")
    ap.add_argument("--out", default="map.html", help="куда сохранить интерактивную карту")
    args = ap.parse_args(argv)

    # ---- 1. получить места
    if args.demo:
        places = list(DEMO_PLACES)
    elif args.places_file:
        places = load_places_file(args.places_file)
    else:
        if not (args.query or args.category):
            ap.error("нужно указать --demo, --places-file, --query или --category")
        center = geocode(args.city, limit=1)
        if not center:
            print(f"Город не найден: {args.city}", file=sys.stderr)
            return 2
        p0 = center[0]
        dlat = args.radius_km * 1000 / 111_320
        dlon = args.radius_km * 1000 / (111_320 * max(0.2, abs(math.cos(math.radians(p0.lat)))))
        bbox = (p0.lat - dlat, p0.lon - dlon, p0.lat + dlat, p0.lon + dlon)
        places = find_places(query=args.query, category=args.category,
                             bbox=bbox, limit=args.limit)
    if not places:
        print("Места не получены — проверьте запрос/сеть/файл.", file=sys.stderr)
        return 1
    print(f"📍 Мест: {len(places)}")

    # ---- 2. кластеризация по плотности
    clusters, noise = cluster_places(places, eps_m=args.eps,
                                     min_samples=args.min_samples, method=args.method)
    print(f"🧩 Кластеров: {len(clusters)}, шум (вне кластеров): {len(noise)}")
    ordered = [p for c in clusters for p in c.members] + noise
    labels = [c.cid for c in clusters for _ in c.members] + [-1] * len(noise)
    sil = silhouette_check(ordered, labels)
    if sil is not None:
        print(f"   Качество разделения (silhouette): {sil:.2f}")
    for c in clusters:
        print(f"   Кластер {c.cid}: {c.size} мест, радиус ~{c.radius_m():.0f} м, "
              f"центр ({c.center[0]:.5f}, {c.center[1]:.5f})")

    # ---- 3. оптимальный маршрут
    depot = parse_coords(args.depot) if args.depot else None
    route = plan_route(clusters, noise=noise, depot=depot,
                       mode=args.mode, include_noise=not args.no_noise)
    print("🛣 " + "\n🛣 ".join(route.summary_lines()))
    for i, s in enumerate(route.stops, start=1):
        cl = f" [кластер {s.cluster_id}]" if s.cluster_id else ""
        print(f"   {i:>3}. {s.name}{cl}")

    # ---- 4. карта
    m = render_map(places, clusters=clusters, noise=noise, route=route,
                   title=f"Кластеры и маршрут ({args.mode})")
    m.save(args.out)
    print(f"💾 Карта сохранена: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
