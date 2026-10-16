"""CLI: поиск мест → кластеризация по плотности → оптимальный маршрут → HTML-карта.

Примеры:
  python cli.py --demo --out map.html

  # места по СПИСКУ АДРЕСОВ (координаты определяются геокодером OSM):
  python cli.py --addresses sample_addresses.txt --country Казань \
                --eps 500 --mode clusters --out addresses_map.html
  python cli.py --address "Кремль, Казань" --address "ул. Баумана, 20, Казань" \
                --address "55.796,49.106" --out two.html

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
import time

from clustering import cluster_places, silhouette_check
from geo_utils import parse_coords
from places import (CATEGORY_TAGS, Place, find_places, geocode, active_geocoder,
                    get_nominatim_min_interval, load_addresses_file,
                    load_places_by_addresses, set_geocoder,
                    set_nominatim_min_interval)
from routing import plan_route
from visualization import render_map


def load_places_file(path: str, max_workers: int = 4) -> list[Place]:
    """Места из CSV/JSON. lat/lon необязательны: если их нет, строка
    воспринимается как адрес и геокодируется (Photon, резерв — Nominatim)."""
    with open(path, encoding="utf-8-sig") as f:
        text = f.read()
    if path.endswith(".json"):
        rows = json.loads(text)
    else:
        import csv
        rows = list(csv.DictReader(io.StringIO(text)))
    out, addr_rows = [], []
    for i, r in enumerate(rows):
        name = str(r.get("name", "")).strip()
        category = str(r.get("category", "")).strip()
        address = str(r.get("address", "")).strip()
        try:
            lat, lon = float(r["lat"]), float(r["lon"])
        except (KeyError, ValueError, TypeError):
            lat = lon = None
        if lat is not None and lon is not None:
            out.append(Place(name or address or f"Место {i+1}", lat, lon,
                             address=address, category=category))
        elif address:
            addr_rows.append((i, name, category, address))
        # строка без координат и без адреса пропускаем
    if addr_rows:
        print(f"🌐 Геокодирую {len(addr_rows)} адрес(ов) из файла… "
              f"(потоков: {max_workers})")
        resolved, failed = load_places_by_addresses(
            [a for _, _, _, a in addr_rows], max_workers=max_workers)
        # load_places_by_addresses дедуплицирует точки, поэтому сопоставляем
        # обратно по исходному адресу из tags["_input_address"]
        by_input = {p.tags.get("_input_address", ""): p for p in resolved}
        added: set[int] = set()
        for (i, name, category, address) in addr_rows:
            p = by_input.get(address)
            if p is None or id(p) in added:
                continue  # не найден или уже добавлен (дубликат адреса в файле)
            added.add(id(p))
            p.name = name or p.name
            p.category = category or p.category
            out.append(p)
        if failed:
            print(f"[warn] не найдено адресов: {len(failed)} → {failed[:5]}")
    return out


def load_single_address(address: str) -> "Place | None":
    """Один адрес → место (обёртка для совместимости интерфейса CLI)."""
    places, _ = load_places_by_addresses([address])
    return places[0] if places else None


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
    src.add_argument("--places-file", help="CSV/JSON со столбцами name,lat,lon[,category,address]; "
                                           "без lat/lon строка считается адресом и геокодируется")
    src.add_argument("--addresses", help="файл .txt/.csv/.json со списком АДРЕСОВ "
                                         "(по одному в строке) — координаты определяются геокодером")
    src.add_argument("--address", action="append", default=[], metavar="АДРЕС",
                    help="адрес места; можно повторять несколько раз "
                         "(--address \"ул. Баумана, 20\" --address ...)")
    ap.add_argument("--country", default="", metavar="ПОДСКАЗКА",
                    help="страна/город для уточнения адресов (напр. «Россия» или «Казань»)")
    ap.add_argument("--replace", action="store_true",
                    help="при импорте из файла/адресов заменить текущий набор мест вместо добавления")
    ap.add_argument("--query", help="текстовый запрос к геокодеру (название/адрес места)")
    ap.add_argument("--category", choices=list(CATEGORY_TAGS), help="категория OSM через Overpass")
    ap.add_argument("--city", default="Москва", help="город для границы поиска")
    ap.add_argument("--radius-km", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--method", default="dbscan", choices=["dbscan", "hdbscan"])
    ap.add_argument("--eps", type=float, default=500.0, help="радиус окрестности, метры")
    ap.add_argument("--min-samples", type=int, default=3)
    ap.add_argument("--mode", default="clusters", choices=["clusters", "places"],
                    help="уровень оптимизации маршрута")
    ap.add_argument("--geocoder", default=None, choices=["photon", "nominatim", "auto"],
                    help="сервис геокодирования адресов: photon (быстрый, по умолчанию), "
                         "nominatim (медленнее, rate-limit 1 запрос/2 с) или auto "
                         "(Photon с автоматическим резервом Nominatim)")
    ap.add_argument("--threads", type=int, default=4, metavar="N",
                    help="число параллельных потоков геокодирования адресов "
                         "(по умолчанию 4; Photon не имеет жёсткого лимита и "
                         "параллелится полностью, Nominatim автоматически "
                         "ограничивается rate-limiter'ом)")
    ap.add_argument("--api-interval", type=float, default=None, metavar="СЕК",
                    help="минимальный интервал между запросами к Nominatim "
                         "(по умолчанию 2 с; актуально только в режимах "
                         "nominatim/auto при сбое Photon)")
    ap.add_argument("--depot", help="старт/финиш 'лат,лон'")
    ap.add_argument("--no-noise", action="store_true", help="не включать места вне кластеров")
    ap.add_argument("--out", default="map.html", help="куда сохранить интерактивную карту")
    args = ap.parse_args(argv)

    if args.geocoder:
        set_geocoder(args.geocoder)
    if args.api_interval is not None:
        set_nominatim_min_interval(args.api_interval)

    # ---- 1. получить места
    places: list[Place] = []
    if args.demo:
        places = list(DEMO_PLACES)
    elif args.address or args.addresses:
        # список адресов: прямо из командной строки и/или из файла
        addr_lines = list(args.address)
        if args.addresses:
            addr_lines += load_addresses_file(args.addresses)
        print(f"🌐 Геокодирую {len(addr_lines)} адрес(ов)… сервис: "
              f"{active_geocoder()}, потоков: {args.threads}")
        t0 = time.perf_counter()
        places, failed = load_places_by_addresses(addr_lines, country_bias=args.country,
                                                  max_workers=args.threads)
        print(f"   готово за {time.perf_counter() - t0:.1f} с")
        for a in failed:
            print(f"[warn] адрес не найден: {a}")
    elif args.places_file:
        places = load_places_file(args.places_file, max_workers=args.threads)
    else:
        if not (args.query or args.category):
            ap.error("нужно указать --demo, --places-file, --addresses/--address, "
                     "--query или --category")
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
    for p in places[:20]:
        src = {"address": "адрес", "coords": "координаты",
               "photon": "Photon", "nominatim": "Nominatim"}.get(p.source, p.source)
        if src:
            print(f"   • {p.label()}  [{src}]")

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
