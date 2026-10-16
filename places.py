"""Поиск мест на карте через OpenStreetMap (Nominatim) и Overpass API.

Всё работает без API-ключей. Nominatim требует вежливого использования:
User-Agent + пауза между запросами (см. https://nominatim.org/release-docs/).

Также здесь — импорт мест по СПИСКУ АДРЕСОВ: пользователь вводит привычные
адреса («ул. Профсоюзов, 15, Казань»), они геокодируются через Nominatim
и превращаются в точки с координатами, готовые к кластеризации.
"""
from __future__ import annotations

import io
import json
import re
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


# ---------------------------------------------------------------- Геокодер
# Кэш запросов к Nominatim (в памяти + на диске), чтобы при повторном
# запуске не дёргать сервис и соблюдать лимит 1 запрос/сек.
GEOCACHE_PATH = ".geocache.json"
_geo_mem_cache: dict[str, list[dict]] | None = None


def _geo_cache_load() -> dict[str, list[dict]]:
    global _geo_mem_cache
    if _geo_mem_cache is None:
        try:
            with open(GEOCACHE_PATH, encoding="utf-8") as f:
                _geo_mem_cache = json.load(f)
        except (OSError, ValueError):
            _geo_mem_cache = {}
    return _geo_mem_cache


def _geo_cache_save() -> None:
    try:
        with open(GEOCACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_geo_mem_cache or {}, f, ensure_ascii=False)
    except OSError:
        pass


def geocode_raw(query: str, limit: int = 10, viewbox: tuple | None = None,
                use_cache: bool = True) -> list[dict]:
    """Сырые результаты Nominatim с кэшированием. Возвращает список dict."""
    key = f"{query}|{limit}|{viewbox}"
    cache = _geo_cache_load()
    if use_cache and key in cache:
        return cache[key]
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
    if use_cache:
        cache[key] = data
        _geo_cache_save()
    time.sleep(1.1)  # вежливость к Nominatim: не чаще 1 запроса в секунду
    return data


def geocode(query: str, limit: int = 10, viewbox: tuple | None = None,
            use_cache: bool = True) -> list[Place]:
    """Геокодирование адреса/названия места через Nominatim.

    viewbox — (lon_min, lat_min, lon_max, lat_max) для предпочтения региона.
    """
    data = geocode_raw(query, limit=limit, viewbox=viewbox, use_cache=use_cache)
    places = []
    for item in data:
        addr = item.get("address", {})
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


# ------------------------------------------------- Импорт мест по адресам
_LINE_SPLIT_RE = re.compile(r"[\r\n;]+")


def parse_addresses(text: str) -> list[str]:
    """Разбивает пользовательский текст на список адресов.

    Принимает «каждый адрес с новой строки», через точку с запятой или
    перенос строки внутри одной строки. Пустые строки отбрасываются.
    """
    return [s.strip() for s in _LINE_SPLIT_RE.split(text or "") if s.strip()]


def geocode_address(address: str, country_bias: str = "",
                    viewbox: tuple | None = None) -> Place | None:
    """Один адрес → одна точка. country_bias дописывается к адресу,
    если в нём ещё не упоминается (улучшает поиск за пределами РФ)."""
    q = address.strip()
    if not q:
        return None
    if country_bias and country_bias.lower() not in q.lower():
        q = f"{q}, {country_bias}"
    candidates = geocode_raw(q, limit=3, viewbox=viewbox)
    if not candidates:
        return None
    item = candidates[0]
    addr = item.get("address", {})
    house = addr.get("house_number", "")
    road = addr.get("road", "")
    short = ", ".join(x for x in (road + ((" " + house) if house else "")).split(", ")
                      if x) or addr.get("city") or address
    return Place(
        name=short or address,
        lat=float(item["lat"]),
        lon=float(item["lon"]),
        address=item.get("display_name", address),
        category=addr.get("amenity", "") or "Адрес",
        source="address",
        tags={**addr, "_input_address": address},
    )


def load_places_by_addresses(lines_or_text: str | list[str],
                             country_bias: str = "",
                             viewbox: tuple | None = None,
                             progress_cb=None) -> tuple[list[Place], list[str]]:
    """Список адресов → места с координатами (геокодирование по адресам).

    Аргументы:
      lines_or_text — строка с адресами (по одному в строке / через ';')
                     либо уже готовый список строк;
      country_bias  — страна/город-подсказка («Россия», «Казань»), добавляется
                     к адресу, если сам адрес неоднозначен;
      viewbox       — необязательная рамка (lon_min, lat_min, lon_max, lat_max),
                     ограничивающая поиск регионом;
      progress_cb(done, total) — колбэк прогресса (для интерфейсов).

    Возвращает (места, список адресов, которые не удалось найти).
    Строки, которые выглядят как «лат, лон», принимаются как координаты
    напрямую — сеть для них не нужна.
    """
    from geo_utils import parse_coords

    if isinstance(lines_or_text, str):
        raw_lines = parse_addresses(lines_or_text)
    else:
        raw_lines = [str(s).strip() for s in lines_or_text if str(s).strip()]

    places: list[Place] = []
    failed: list[str] = []
    seen: set[tuple[int, int]] = set()
    total = len(raw_lines)
    for i, line in enumerate(raw_lines):
        pc = parse_coords(line)
        if pc:
            p = Place(name=f"Точка {line}", lat=pc[0], lon=pc[1],
                      address=line, category="Координаты", source="coords")
        else:
            try:
                p = geocode_address(line, country_bias=country_bias, viewbox=viewbox)
            except Exception as exc:
                print(f"[warn] геокодер ({line}): {exc}")
                p = None
        if p is None:
            failed.append(line)
        else:
            key = (round(p.lat * 10000), round(p.lon * 10000))
            if key not in seen:          # дубликаты адресов -> один раз
                seen.add(key)
                places.append(p)
        if progress_cb:
            progress_cb(i + 1, total)
    return places, failed


def load_addresses_file(path: str) -> list[str]:
    """Читает список адресов из .txt/.csv/.tsv/.json.

      .txt/.tsv — по одному адресу в строке;
      .csv      — столбец 'address' (или первая колонка);
      .json     — список строк либо список объектов {"address": ...}.
    """
    with open(path, encoding="utf-8-sig") as f:
        text = f.read()
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    if ext == "json":
        data = json.loads(text)
        out = []
        for r in data if isinstance(data, list) else []:
            if isinstance(r, str):
                out.append(r)
            elif isinstance(r, dict) and r.get("address"):
                out.append(str(r["address"]))
        return [s.strip() for s in out if s.strip()]
    if ext == "csv":
        import csv
        rows = list(csv.reader(io.StringIO(text)))
        if not rows:
            return []
        header = [h.strip().lower() for h in rows[0]]
        idx = header.index("address") if "address" in header else 0
        return [r[idx].strip() for r in rows[1:] if len(r) > idx and r[idx].strip()]
    # txt / tsv / без расширения — каждая непустая строка является адресом
    return parse_addresses(text)


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
