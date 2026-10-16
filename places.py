"""Поиск мест на карте через открытые геокодеры и Overpass API (OpenStreetMap).

ГЕОКОДИРОВАНИЕ АДРЕСОВ — через сервис Photon (https://photon.komoot.io):
быстрый открытый геокодер по данным OSM, НЕ требующий API-ключа и жёсткого
rate-limit, поэтому запросы можно безопасно параллелить потоками. Если
Photon недоступен, программа автоматически переключается на резервный
сервис Nominatim (он медленнее — ограничен 1 запросом в 2 секунды).

Выбор сервиса:  places.set_geocoder("photon" | "nominatim" | "auto")
или переменная окружения GEOCODER=photon|nominatim|auto.

Также здесь — импорт мест по СПИСКУ АДРЕСОВ: пользователь вводит привычные
адреса («ул. Профсоюзов, 15, Казань»), они геокодируются и превращаются
в точки с координатами, готовые к кластеризации.
"""
from __future__ import annotations

import io
import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

USER_AGENT = "map-cluster-router/1.0 (educational demo; contact: local)"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
PHOTON_URL = "https://photon.komoot.io/api/"
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


# ---------------------------------------------------------------- Геокодер
# Кэш запросов к геокодерам (в памяти + на диске), чтобы при повторном
# запуске не дёргать сервис. Доступ из нескольких потоков защищён локом.
GEOCACHE_PATH = ".geocache.json"
_geo_mem_cache: dict[str, list[dict]] | None = None
_geo_cache_lock = threading.Lock()


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


class _RateLimiter:
    """Простой ограничитель частоты обращений к API (потокобезопасный).

    Гарантирует, что сетевые запросы уходят не чаще одного раза в
    `min_interval` секунд, распределяя их равномерно между всеми
    потоками геокодирования.
    """

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def acquire(self) -> None:
        """Заблокировать вызывающий поток до разрешённого момента."""
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self.min_interval
        wait = slot - time.monotonic()
        if wait > 0:
            time.sleep(wait)


# Публичный лимитер для Nominatim. По умолчанию — 1 запрос/2 сек
# (требование вежливости публичного сервиса). Значение можно менять:
#   places.NOMINATIM_MIN_INTERVAL = 0.5
NOMINATIM_MIN_INTERVAL = 2.0
_nominatim_limiter = _RateLimiter(NOMINATIM_MIN_INTERVAL)


def get_nominatim_min_interval() -> float:
    """Текущий минимальный интервал между реальными запросами к Nominatim."""
    return _nominatim_limiter.min_interval


def set_nominatim_min_interval(seconds: float) -> None:
    """Изменить минимальный интервал между запросами (сек)."""
    _nominatim_limiter.min_interval = max(0.0, float(seconds))


# ------------------------------------------------- Выбор сервиса-геокодера
# Основной сервис — Photon (komoot): быстрый, без ключей, можно параллелить.
# Резервный — Nominatim (OSM): медленнее из-за правила 1 запрос / 2 секунды.
#   "auto"      — сначала Photon, при сетевых сбоях автоматический переход
#                 на Nominatim (режим по умолчанию);
#   "photon"    — только Photon;
#   "nominatim" — только Nominatim.
GEOCODER_AUTO = "auto"
GEOCODER_PHOTON = "photon"
GEOCODER_NOMINATIM = "nominatim"
_VALID_GEOCODERS = (GEOCODER_AUTO, GEOCODER_PHOTON, GEOCODER_NOMINATIM)

_geocoder_mode = os.environ.get("GEOCODER", GEOCODER_AUTO).strip().lower()
if _geocoder_mode not in _VALID_GEOCODERS:
    _geocoder_mode = GEOCODER_AUTO
_photon_failures = 0          # подряд неудачных сетевых попыток Photon
_photon_state_lock = threading.Lock()
PHOTON_FAILURE_LIMIT = 3      # после этого «auto» переключается на Nominatim


def set_geocoder(mode: str) -> None:
    """Выбрать геокодер: 'photon' | 'nominatim' | 'auto' (Photon с резервом)."""
    global _geocoder_mode, _photon_failures
    mode = (mode or "").strip().lower()
    if mode not in _VALID_GEOCODERS:
        raise ValueError(f"Неизвестный геокодер: {mode!r}. "
                         f"Допустимо: {', '.join(_VALID_GEOCODERS)}")
    with _photon_state_lock:
        _geocoder_mode = mode
        _photon_failures = 0  # сбрасываем счётчик сбоев при явном выборе


def get_geocoder() -> str:
    """Текущий режим выбора геокодера ('photon' | 'nominatim' | 'auto')."""
    return _geocoder_mode


def active_geocoder() -> str:
    """Реально используемый сейчас сервис ('photon' или 'nominatim')."""
    if _geocoder_mode == GEOCODER_NOMINATIM:
        return GEOCODER_NOMINATIM
    with _photon_state_lock:
        if _geocoder_mode == GEOCODER_AUTO and _photon_failures >= PHOTON_FAILURE_LIMIT:
            return GEOCODER_NOMINATIM
    return GEOCODER_PHOTON


def _http_get_json(url: str, params: dict | None = None,
                   timeout: int = 30, retries: int = 1) -> dict | list:
    """GET c разбором JSON и простыми повторными попытками (5xx/сеть)."""
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code < 500:      # 4xx повторять бессмысленно
                raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc
        if attempt < retries:
            time.sleep(0.4 * (attempt + 1))
    raise last_exc  # type: ignore[misc]


# ------------------------------------------------------------------ Photon
_PHOTON_TYPE_RU = {
    "country": "Страна", "state": "Регион", "city": "Город",
    "town": "Город", "village": "Населённый пункт", "municipality": "Муниципалитет",
    "district": "Район", "suburb": "Район", "locality": "Поселение",
    "street": "Улица", "pedestrian": "Улица", "avenue": "Проспект",
    "road": "Дорога", "residential": "Улица", "unclassified": "Дорога",
    "housenumber": "Дом", "building": "Здание", "house": "Дом",
    "amenity": "Объект", "shop": "Магазин", "tourism": "Достопримечательность",
    "leisure": "Отдых", "historic": "Историческое место", "nature": "Природа",
    "other": "",
}


def _photon_display_name(props: dict) -> str:
    """Собираем человекочитаемый адрес из GeoJSON-свойств Photon."""
    name = props.get("name", "")
    street = props.get("street", "")
    hn = props.get("housenumber", "")
    parts: list[str] = []
    if street:
        parts.append(street + (f", {hn}" if hn else ""))
    elif name and not hn:
        parts.append(name)
    if hn and (street or name):
        if not street:
            parts[0] = f"{name}, {hn}" if name else hn
    place = props.get("city_district") or props.get("suburb") \
        or props.get("neighbourhood") or props.get("district")
    if place:
        parts.append(place)
    for key in ("city", "town", "village"):
        v = props.get(key)
        if v and v not in parts:
            parts.append(v)
            break
    # регион/страна — если города нет, но есть более крупный объект
    if not any(props.get(k) for k in ("city", "town", "village")):
        for key in ("county", "state", "country"):
            v = props.get(key)
            if v and v not in parts:
                parts.append(v)
                break
    return ", ".join(p for p in parts if p)


def _photon_relevance(item: dict, query: str) -> float:
    """Эвристика релевантности результата Photon для запроса «адрес, город».

    Возвращает штраф (меньше — лучше). Штрафуется выбор объекта из другого
    населённого пункта, чем указан в запросе (защита от «Казань → Казанлък»).
    """
    addr = item.get("address", {})
    q = (query or "").lower()
    penalty = 0.0
    for place_key in ("city", "town", "village", "county", "state", "country"):
        v = str(addr.get(place_key, "")).strip().lower()
        if not v:
            continue
        tokens = [t for t in re.split(r"[^\w]+", v, flags=re.UNICODE) if len(t) > 3]
        if tokens and not any(t in q for t in tokens):
            penalty += 1.0   # объект не из указанного в запросе места
        break
    if not item.get("name"):
        penalty += 0.2       # безымянные чуть менее ценны, чем именованные POI
    return penalty


def _photon_to_items(data: dict, limit: int, query: str = "") -> list[dict]:
    """GeoJSON Photon → список словарей в формате Nominatim (единый вид).

    Результаты ранжируются: сначала совпадение с населённым пунктом из
    запроса (см. _photon_relevance), затем — порядок самого Photon.
    """
    items: list[dict] = []
    for feat in data.get("features") or []:
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            continue
        lon, lat = coords[0], coords[1]
        props = feat.get("properties") or {}
        addr = {k: props[k] for k in
                ("street", "housenumber", "city", "town", "village", "state",
                 "country", "postcode", "district", "suburb", "neighbourhood",
                 "county", "city_district") if props.get(k)}
        otype = props.get("type", "") or props.get("osm_key", "")
        category = _PHOTON_TYPE_RU.get(otype, props.get("osm_value", otype))
        disp = _photon_display_name(props) or props.get("name", "")
        items.append({
            "lat": str(lat), "lon": str(lon),
            "display_name": disp,
            "name": props.get("name", ""),
            "category": category,
            "type": otype,
            "address": addr,
        })
    # стабильная сортировка: штраф релевантности, порядок от Photon сохраняем
    scored = [(_photon_relevance(it, query), i, it) for i, it in enumerate(items)]
    scored.sort(key=lambda t: (t[0], t[1]))
    return [it for _, _, it in scored[:limit]]


def _geocode_photon(query: str, limit: int, viewbox: tuple | None) -> list[dict]:
    global _photon_failures
    params = {"q": query, "limit": limit}
    if viewbox:
        # Photon принимает bbox как lon_min,lat_min,lon_max,lat_max
        params["bbox"] = ",".join(str(v) for v in viewbox)
    try:
        data = _http_get_json(PHOTON_URL, params, timeout=20, retries=1)
    except Exception:
        with _photon_state_lock:
            _photon_failures += 1
        raise
    with _photon_state_lock:
        _photon_failures = 0     # успех — сбрасываем счётчик сбоев
    return _photon_to_items(data, limit, query)


def _geocode_nominatim(query: str, limit: int, viewbox: tuple | None) -> list[dict]:
    _nominatim_limiter.acquire()  # пауза ДО запроса, равномерный темп
    params = {
        "q": query,
        "format": "jsonv2",
        "limit": limit,
        "addressdetails": 1,
    }
    if viewbox:
        params["viewbox"] = ",".join(str(v) for v in viewbox)
        params["bounded"] = 1
    return _http_get_json(NOMINATIM_URL, params)


def geocode_raw(query: str, limit: int = 10, viewbox: tuple | None = None,
                use_cache: bool = True) -> list[dict]:
    """Сырые результаты геокодирования с кэшированием и выбором сервиса.

    По умолчанию работает «Photon, при проблемах — Nominatim». Результат
    всегда приводится к единому формату Nominatim ({lat, lon, display_name,
    address, ...}), поэтому остальной код не зависит от выбранного сервиса.

    Потокобезопасен: несколько потоков могут вызывать его параллельно.
    Кэш ускоряет повторные запросы; лимитер активен только для Nominatim.
    """
    key = f"{query}|{limit}|{viewbox}"
    with _geo_cache_lock:
        cache = _geo_cache_load()
        if use_cache and key in cache:
            return cache[key]

    mode = _geocoder_mode
    want_photon = mode in (GEOCODER_AUTO, GEOCODER_PHOTON)
    want_nom = mode in (GEOCODER_AUTO, GEOCODER_NOMINATIM)
    data: list[dict] | None = None
    photon_error: Exception | None = None

    if want_photon and active_geocoder() == GEOCODER_PHOTON:
        try:
            data = _geocode_photon(query, limit, viewbox)
        except Exception as exc:
            photon_error = exc
            data = None
            if not want_nom:
                raise

    if data is None and want_nom:
        if mode == GEOCODER_AUTO and photon_error is not None:
            print(f"[info] Photon недоступен ({photon_error}), переключаюсь на Nominatim…")
        data = _geocode_nominatim(query, limit, viewbox)

    if data is None:
        data = []
    if use_cache:
        with _geo_cache_lock:
            cache[key] = data
            _geo_cache_save()
    return data


def geocode(query: str, limit: int = 10, viewbox: tuple | None = None,
            use_cache: bool = True) -> list[Place]:
    """Геокодирование адреса/названия места (Photon, резерв — Nominatim).

    viewbox — (lon_min, lat_min, lon_max, lat_max) для предпочтения региона.
    """
    data = geocode_raw(query, limit=limit, viewbox=viewbox, use_cache=use_cache)
    places = []
    for item in data:
        addr = item.get("address", {})
        places.append(Place(
            name=item.get("name") or addr.get("road") or addr.get("street") or query,
            lat=float(item["lat"]),
            lon=float(item["lon"]),
            address=item.get("display_name", ""),
            category=item.get("category", "") or item.get("type", ""),
            source=active_geocoder(),
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
    candidates = geocode_raw(q, limit=5, viewbox=viewbox)
    if not candidates:
        return None
    item = candidates[0]
    addr = item.get("address", {})
    house = addr.get("house_number", "") or addr.get("housenumber", "")
    road = addr.get("road", "") or addr.get("street", "")
    short = ", ".join(x for x in (road + (((" " + house) if house else ""))
                      ).split(", ") if x) or addr.get("city") or item.get("name") \
        or addr.get("locality") or address
    return Place(
        name=short or address,
        lat=float(item["lat"]),
        lon=float(item["lon"]),
        address=item.get("display_name", address),
        category=item.get("category", "") or addr.get("amenity", "") or "Адрес",
        source=active_geocoder(),
        tags={**addr, "_input_address": address},
    )


def load_places_by_addresses(lines_or_text: str | list[str],
                             country_bias: str = "",
                             viewbox: tuple | None = None,
                             progress_cb=None,
                             max_workers: int = 4) -> tuple[list[Place], list[str]]:
    """Список адресов → места с координатами (геокодирование по адресам).

    Аргументы:
      lines_or_text — строка с адресами (по одному в строке / через ';')
                     либо уже готовый список строк;
      country_bias  — страна/город-подсказка («Россия», «Казань»), добавляется
                     к адресу, если сам адрес неоднозначен;
      viewbox       — необязательная рамка (lon_min, lat_min, lon_max, lat_max),
                     ограничивающая поиск регионом;
      progress_cb(done, total) — колбэк прогресса (для интерфейсов);
      max_workers   — число параллельных потоков геокодирования. Основной
                     геокодер Photon не имеет жёсткого лимита запросов, поэтому
                     параллельность даёт почти линейное ускорение. Если включён
                     режим Nominatim (или сработал автоматический резерв), общий
                     темп ограничивается rate-limiter'ом (NOMINATIM_MIN_INTERVAL),
                     и правила вежливости сервиса соблюдаются автоматически.
                     Кэшированные адреса возвращаются мгновенно, без сети.

    Возвращает (места, список адресов, которые не удалось найти).
    Строки, которые выглядят как «лат, лон», принимаются как координаты
    напрямую — сеть для них не нужна. Порядок мест соответствует порядку
    адресов во входном списке.
    """
    from geo_utils import parse_coords

    if isinstance(lines_or_text, str):
        raw_lines = parse_addresses(lines_or_text)
    else:
        raw_lines = [str(s).strip() for s in lines_or_text if str(s).strip()]

    total = len(raw_lines)
    results: list[Place | None] = [None] * total

    def _resolve(idx: int, line: str) -> Place | None:
        pc = parse_coords(line)
        if pc:  # координаты — без сети
            return Place(name=f"Точка {line}", lat=pc[0], lon=pc[1],
                        address=line, category="Координаты", source="coords")
        try:
            return geocode_address(line, country_bias=country_bias, viewbox=viewbox)
        except Exception as exc:
            print(f"[warn] геокодер ({line}): {exc}")
            return None

    done = 0
    lock = threading.Lock()

    def _bump() -> None:
        nonlocal done
        with lock:
            done += 1
            if progress_cb:
                progress_cb(done, total)

    workers = max(1, min(int(max_workers), total)) if total else 1
    if workers == 1:
        for i, line in enumerate(raw_lines):
            results[i] = _resolve(i, line)
            _bump()
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_resolve, i, line): i
                       for i, line in enumerate(raw_lines)}
            for fut in as_completed(futures):
                i = futures[fut]
                results[i] = fut.result()
                _bump()

    places: list[Place] = []
    failed: list[str] = []
    seen: set[tuple[int, int]] = set()
    for line, p in zip(raw_lines, results):
        if p is None:
            failed.append(line)
            continue
        key = (round(p.lat * 10000), round(p.lon * 10000))
        if key not in seen:          # дубликаты адресов -> один раз
            seen.add(key)
            places.append(p)
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
    """Поиск всех мест категории в прямоугольнике bbox=(south, west, north, east) через Overpass.

    Каждый OSM-тег категории (например cafe/restaurant/fast_food) запрашивается
    отдельным параллельным запросом — так поиск по много-теговым категориям
    заметно быстрее. Если все параллельные запросы упали, выполняется одна
    общая попытка комбинированным запросом.
    """
    tags = CATEGORY_TAGS.get(category)
    if not tags:
        raise ValueError(f"Неизвестная категория: {category}")

    def _run_query(query: str) -> list[dict]:
        req = urllib.request.Request(
            OVERPASS_URL,
            data=("data=" + urllib.parse.quote(query)).encode(),
            headers={"User-Agent": USER_AGENT, "Content-Type": "application/x-www-form-urlencoded"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("elements", [])

    def _selectors_for(tag_list: list[dict[str, str]]) -> str:
        selectors = []
        for t in tag_list:
            key, val = next(iter(t.items()))
            if val:
                selectors.append(f'nwr["{key}"="{val}"]{_overpass_bbox(bbox)};')
            else:
                selectors.append(f'nwr["{key}"]{_overpass_bbox(bbox)};')
        return "".join(selectors)

    def _fetch(tag_list: list[dict[str, str]]) -> list[dict]:
        query = ("[out:json][timeout:30];(" + _selectors_for(tag_list) +
                 ");out center " + str(limit) + ";")
        return _run_query(query)

    elements: list[dict] = []
    errors: list[str] = []
    if len(tags) == 1:
        try:
            elements.extend(_fetch(tags))
        except Exception as exc:
            errors.append(str(exc))
    else:
        # По одному запросу на тег — выполняем параллельно (I/O-bound).
        with ThreadPoolExecutor(max_workers=min(len(tags), 4)) as pool:
            futs = [pool.submit(_fetch, [t]) for t in tags]
            for fut in futs:
                try:
                    elements.extend(fut.result())
                except Exception as exc:
                    errors.append(str(exc))
        # Если параллельные запросы не дали ничего — одна общая попытка
        if not elements and errors:
            try:
                elements = _fetch(tags)
                errors = []
            except Exception as exc:
                errors.append(str(exc))
    if errors and not elements:
        raise RuntimeError("; ".join(errors[:2]))

    places = []
    for el in elements:
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
                limit: int = 50, delay_s: float = 0.0) -> list[Place]:
    """Комбинированный поиск: текстовый запрос (геокодер) и/или категория в регионе (Overpass).

    Геокодер (Photon/Nominatim) и Overpass — разные сервисы, поэтому запросы
    к ним уходят параллельно из двух потоков. Возвращает список мест без
    дубликатов по координатам.
    """
    tasks = []
    if category and bbox:
        tasks.append(("overpass", lambda: search_category(category, bbox, limit=limit)))
    if query:
        vb = None
        if bbox:
            s, w, n, e = bbox
            vb = (w, s, e, n)

        def _geo(q=query, vb=vb, lim=limit):
            # geocode_raw сам выбирает сервис (Photon → резерв Nominatim)
            return geocode(q, limit=min(lim, 20), viewbox=vb)
        tasks.append(("geocoder", _geo))

    places: list[Place] = []
    if len(tasks) == 1:
        kind, fn = tasks[0]
        try:
            places.extend(fn())
        except Exception as exc:
            print(f"[warn] {kind}: {exc}")
    elif len(tasks) > 1:
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futs = [(kind, pool.submit(fn)) for kind, fn in tasks]
            for kind, fut in futs:
                try:
                    places.extend(fut.result())
                except Exception as exc:
                    print(f"[warn] {kind}: {exc}")
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
