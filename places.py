"""Поиск мест на карте через OpenStreetMap (Nominatim) и Overpass API.

Всё работает без API-ключей. Nominatim требует вежливого использования:
User-Agent + пауза между запросами (см. https://nominatim.org/release-docs/).
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

USER_AGENT = "map-cluster-router/1.0 (educational demo; contact: local)"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Категории мест для Overpass (OSM-теги)
CATEGORY_TAGS: dict[str, list[dict[str, str]]] = {
    "Кафе и рестораны": [{"amenity": "cafe"}, {"amenity": "restaurant"}, {"amenity": "fast_food"}],
    "Достопримечательности": [{"tourism": "attraction"}, {"tourism": "artwork"}, {"historic": ""}],
    "Парки": [{"leisure": "park"}, {"leisure": "garden"}],
    "Музеи": [{"tourism": "museum"}, {"tourism": "gallery"}],
    "Магазины": [{"shop": "supermarket"}, {"shop": "convenience"}, {"amenity": "marketplace"}],
    "Банки и банкоматы": [{"amenity": "bank"}, {"amenity": "atm"}],
    "Аптеки": [{"amenity": "pharmacy"}],
    "Остановки транспорта": [{"highway": "bus_stop"}, {"public_transport": "platform"}],
}


@dataclass
class Place:
    """Место на карте."""
    name: str
    lat: float
    lon: float
    address: str = ""
    category: str = ""
    source: str = ""           # nominatim / overpass / manual
    tags: dict = field(default_factory=dict)

    @property
    def coords(self) -> tuple[float, float]:
        return self.lat, self.lon

    def label(self) -> str:
        return self.name or self.address or f"{self.lat:.5f}, {self.lon:.5f}"


def _http_get(url: str, params: dict | None = None, timeout: int = 30) -> dict | list:
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def geocode(query: str, limit: int = 10, viewbox: tuple | None = None) -> list[Place]:
    """Геокодирование адреса/названия места через Nominatim.

    viewbox — (lon_min, lat_min, lon_max, lat_max) для предпочтения региона.
    """
    params = {
        "q": query,
        "format": "jsonv2",
        "limit": limit,
        "addressdetails": 1,
    }
    if viewbox:
        params["viewbox"] = ",".join(str(v) for v in viewbox)
        params["bounded"] = 1
    data = _http_get(NOMINATIM_URL, params)
    places = []
    for item in data:
        addr = item.get("address", {})
        parts = [addr.get(k) for k in ("road", "suburb", "city", "state", "country")]
        places.append(Place(
            name=item.get("name") or addr.get("road") or query,
            lat=float(item["lat"]),
            lon=float(item["lon"]),
            address=item.get("display_name", ""),
            category=item.get("category", "") or item.get("type", ""),
            source="nominatim",
            tags=addr,
        ))
    return places


def _overpass_bbox(bbox: tuple[float, float, float, float]) -> str:
    s, w, n, e = bbox  # south, west, north, east
    return f"({s},{w},{n},{e})"


def search_category(category: str, bbox: tuple[float, float, float, float],
                    limit: int = 300) -> list[Place]:
    """Поиск всех мест категории в прямоугольнике bbox=(south, west, north, east) через Overpass."""
    tags = CATEGORY_TAGS.get(category)
    if not tags:
        raise ValueError(f"Неизвестная категория: {category}")
    selectors = []
    for t in tags:
        key, val = next(iter(t.items()))
        if val:
            selectors.append(f'nwr["{key}"="{val}"]{ _overpass_bbox(bbox) };')
        else:
            selectors.append(f'nwr["{key}"]{ _overpass_bbox(bbox) };')
    query = "[out:json][timeout:30];(" + "".join(selectors) + ");out center " + str(limit) + ";"
    req = urllib.request.Request(
        OVERPASS_URL,
        data=("data=" + urllib.parse.quote(query)).encode(),
        headers={"User-Agent": USER_AGENT, "Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    places = []
    for el in data.get("elements", []):
        tags_el = el.get("tags", {})
        if el["type"] == "node":
            lat, lon = el["lat"], el["lon"]
        else:  # way/relation — используем центр
            c = el.get("center")
            if not c:
                continue
            lat, lon = c["lat"], c["lon"]
        places.append(Place(
            name=tags_el.get("name", ""),
            lat=lat,
            lon=lon,
            address=tags_el.get("addr:street", "") or category,
            category=category,
            source="overpass",
            tags=tags_el,
        ))
    # места без имени менее полезны — ставим после именованных
    places.sort(key=lambda p: (p.name == "",))
    return places[:limit]


def find_places(query: str | None = None, category: str | None = None,
                bbox: tuple[float, float, float, float] | None = None,
                limit: int = 50, delay_s: float = 1.1) -> list[Place]:
    """Комбинированный поиск: текстовый запрос (Nominatim) и/или категория в регионе (Overpass).

    Возвращает список мест без дубликатов по координатам.
    """
    places: list[Place] = []
    if category and bbox:
        try:
            places.extend(search_category(category, bbox, limit=limit))
        except Exception as exc:  # сеть может быть недоступна
            print(f"[warn] Overpass ({category}): {exc}")
        time.sleep(delay_s)
    if query:
        try:
            vb = None
            if bbox:
                s, w, n, e = bbox
                vb = (w, s, e, n)
            places.extend(geocode(query, limit=min(limit, 20), viewbox=vb))
        except Exception as exc:
            print(f"[warn] Nominatim ({query}): {exc}")
        time.sleep(delay_s)
    # дедупликация: одна точка ~ в пределах 30 м
    uniq: list[Place] = []
    seen: set[tuple[int, int]] = set()
    for p in places:
        key = (round(p.lat * 10000), round(p.lon * 10000))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq
