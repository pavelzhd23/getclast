"""Веб-интерфейс: поиск мест, кластеризация по плотности, оптимальный маршрут.

Запуск:  streamlit run app.py
Требуется Python 3.10+, а также: folium, scikit-learn, streamlit (опционально).
"""
from __future__ import annotations

import io
import json

import pandas as pd
import streamlit as st

import places as places_mod
from clustering import cluster_places, silhouette_check
from geo_utils import parse_coords, path_length_m
from places import (CATEGORY_TAGS, Place, find_places, geocode,
                    load_places_by_addresses)
from routing import plan_route
from visualization import render_map

st.set_page_config(page_title="Кластеры и маршруты на карте", page_icon="🗺️", layout="wide")
st.title("🗺️ Кластеризация мест на карте и оптимальный маршрут")
st.caption("Находит места (OpenStreetMap), объединяет их в кластеры по плотности "
           "(DBSCAN/HDBSCAN) и строит оптимальный маршрут — по отдельным местам или "
           "по кластерам (двухуровневая TSP-эвристика: nearest-neighbor + 2-opt + branch&bound).")

# --------------------------------------------------------------- 1. Поиск мест
st.header("1️⃣ Места")
tab_search, tab_addr, tab_manual, tab_file = st.tabs(
    ["🔎 Поиск на карте", "📋 По списку адресов", "✍️ Вручную", "📄 Из файла"])

if "places" not in st.session_state:
    # стартовый демо-набор: известные точки Санкт-Петербурга
    st.session_state.places = [
        Place("Эрмитаж", 59.9398, 30.3146, category="Музеи"),
        Place("Русский музей", 59.9385, 30.3370, category="Музеи"),
        Place("Спас на Крови", 59.9407, 30.3320, category="Достопримечательности"),
        Place("Казанский собор", 59.9358, 30.3264, category="Достопримечательности"),
        Place("Исаакиевский собор", 59.9304, 30.1939, category="Достопримечательности"),
        Place("Петропавловская крепость", 59.9504, 30.3217, category="Достопримечательности"),
        Place("Стрелка Васильевского острова", 59.9466, 30.2470, category="Достопримечательности"),
        Place("Аврора", 59.9483, 30.2407, category="Достопримечательности"),
        Place("Новый Голландия", 59.9342, 30.2899, category="Парки"),
        Place("Марсово поле", 59.9411, 30.3378, category="Парки"),
        Place("Летний сад", 59.9436, 30.3356, category="Парки"),
        Place("Гостиный двор", 59.9375, 30.3386, category="Магазины"),
        Place("Пятёрочка наб. Фонтанки", 59.9423, 30.3472, category="Магазины"),
        Place("Кофейня на Невском 48", 59.9356, 30.3310, category="Кафе и рестораны"),
    ]

with tab_search:
    c1, c2 = st.columns([2, 1])
    with c1:
        query = st.text_input("Текстовый запрос (адрес/название)",
                              placeholder="например: Кафе на Невском проспекте")
    with c2:
        category = st.selectbox("Категория (Overpass/OSM)", ["—"] + list(CATEGORY_TAGS))
    city = st.text_input("Город/регион для границы поиска", value="Санкт-Петербург")
    limit = st.slider("Максимум мест", 10, 200, 60)

    if st.button("🔎 Искать места", type="primary"):
        try:
            bbox_res = geocode(city, limit=1) or geocode(query or city, limit=1)
            if not bbox_res:
                st.error("Не удалось определить регион.")
            else:
                p0 = bbox_res[0]
                # рамка ~±4 км вокруг найденного центра города
                dlat = 4000 / 111_320
                dlon = 4000 / (111_320 * max(0.2, abs(__import__("math").cos(__import__("math").radians(p0.lat)))))
                bbox = (p0.lat - dlat, p0.lon - dlon, p0.lat + dlat, p0.lon + dlon)
                found = find_places(query=query or None,
                                    category=None if category == "—" else category,
                                    bbox=bbox, limit=limit)
                if found:
                    st.session_state.places = found
                    st.success(f"Найдено мест: {len(found)}")
                else:
                    st.warning("Ничего не найдено — проверьте запрос или категорию.")
        except Exception as exc:
            st.error(f"Ошибка поиска (нужен доступ в интернет): {exc}")

with tab_addr:
    st.markdown("Введите **список адресов** — по одному в строке (координаты программа "
                "определит сама через геокодер Photon по данным OpenStreetMap). Затем места будут "
                "сгруппированы в кластеры по плотности и построен маршрут.")
    addr_text = st.text_area(
        "Адреса (каждый с новой строки или через «;»)",
        height=200,
        placeholder="ул. Баумана, 20, Казань\nул. Профсоюзов, 15, Казань\n"
                    "Кремль, Казань\nул. Петербургская, 1",
        key="addr_text")
    a1, a2 = st.columns([2, 1])
    with a1:
        country_bias = st.text_input(
            "Подсказка: город/страна (добавляется к коротким адресам)",
            value="", placeholder="например: Казань или Россия", key="addr_bias")
        geo_choice = st.selectbox(
            "Сервис геокодирования",
            ["photon", "auto", "nominatim"], index=0,
            help="Photon — быстрый открытый геокодер по данным OSM (по умолчанию, "
                 "параллелится без жёстких ограничений). Nominatim — резервный "
                 "сервис OSM, медленнее (правило: не чаще 1 запроса в 2 с). "
                 "«auto» — Photon с автоматическим переходом на Nominatim при сбоях.")
        places_mod.set_geocoder(geo_choice)
        n_threads = st.slider(
            "Параллельных потоков геокодирования", 1, 8, 4,
            help="С сервисом Photon параллельные запросы идут на полной скорости. "
                 "Если выбран Nominatim (или сработал авто-резерв), темп "
                 "автоматически ограничивается правилом вежливости сервиса. "
                 "Кэшированные адреса обрабатываются мгновенно.")
    with a2:
        st.write("")
        append_mode = st.checkbox("Добавить к текущим местам", value=True,
                                  help="Снять, чтобы заменить весь набор мест")
    if st.button("🌐 Определить координаты и добавить", type="primary", key="btn_addr"):
        lines = places_mod.parse_addresses(addr_text)
        if not lines:
            st.warning("Список адресов пуст.")
        else:
            bar = st.progress(0.0, text=f"Геокодирую 0/{len(lines)}…")

            def _prog(done, total):
                bar.progress(done / total, text=f"Геокодирую {done}/{total}…")

            try:
                found, failed = load_places_by_addresses(
                    lines, country_bias=country_bias, progress_cb=_prog,
                    max_workers=n_threads)
            except Exception as exc:
                st.error(f"Ошибка геокодирования (нужен доступ в интернет): {exc}")
                found, failed = [], []
            bar.empty()
            if found:
                if append_mode:
                    # не дублировать точки, уже присутствующие в наборе
                    existing = {(round(p.lat * 10000), round(p.lon * 10000))
                                for p in st.session_state.places}
                    new = [p for p in found
                           if (round(p.lat * 10000), round(p.lon * 10000)) not in existing]
                    st.session_state.places.extend(new)
                    st.success(f"Добавлено мест из адресов: {len(new)} "
                               f"(всего в наборе: {len(st.session_state.places)})")
                else:
                    st.session_state.places = found
                    st.success(f"Мест по адресам: {len(found)}")
            if failed:
                st.warning("Не удалось найти адреса:\n" + "\n".join(f"• {a}" for a in failed))

with tab_manual:
    c_m1, c_m2 = st.columns(2)
    with c_m1:
        name = st.text_input("Название места", key="m_name")
        m_address = st.text_input("Или адрес (без координат)", key="m_address",
                                  placeholder="например: ул. Баумана, 30, Казань")
    with c_m2:
        coords_s = st.text_input("Координаты «широта, долгота» (если известны)", key="m_coords",
                                 placeholder="59.9398, 30.3146")
        m_cat = st.text_input("Категория (необязательно)", key="m_cat", placeholder="Кафе")
    if st.button("➕ Добавить место"):
        pc = parse_coords(coords_s)
        if pc:
            st.session_state.places.append(
                Place(name or m_address or f"Точка {len(st.session_state.places)+1}",
                      pc[0], pc[1], address=m_address, category=m_cat, source="manual"))
            st.success("Добавлено по координатам.")
        elif m_address.strip():
            with st.spinner(f"Геокодирую адрес «{m_address}»…"):
                try:
                    found, failed = load_places_by_addresses([m_address], country_bias="")
                except Exception as exc:
                    st.error(f"Ошибка геокодирования: {exc}")
                    found, failed = [], []
            if found:
                p = found[0]
                p.name = name or p.name
                p.category = m_cat or p.category
                st.session_state.places.append(p)
                st.success(f"Добавлено: {p.address}")
            else:
                st.error("Адрес не найден. Уточните адрес или укажите координаты вручную.")
        else:
            st.error("Укажите координаты либо адрес.")

with tab_file:
    upl = st.file_uploader("CSV/JSON со столбцами name,lat,lon[,category,address] "
                           "или .txt/.csv со столбцом address (без координат)",
                           type=["csv", "json", "txt"])
    f_mode_replace = st.checkbox("Заменить текущий набор мест", value=True)
    if upl is not None and st.button("📥 Загрузить"):
        data = upl.read().decode("utf-8-sig")
        loaded: list[Place] = []
        try:
            if upl.name.endswith(".json"):
                for r in json.loads(data):
                    lat, lon = r.get("lat"), r.get("lon")
                    addr = str(r.get("address", "")).strip()
                    if lat is not None and lon is not None:
                        loaded.append(Place(str(r.get("name", "")), float(lat), float(lon),
                                            address=addr, category=str(r.get("category", ""))))
                    elif addr:
                        loaded.append(Place("", 0.0, 0.0, address=addr,
                                            category=str(r.get("category", ""))))
            elif upl.name.endswith(".txt"):
                # простой список адресов: по одному в строке
                found, failed = load_places_by_addresses(data, country_bias=country_bias)
                loaded = found
                if failed:
                    st.warning("Не найдено адресов: " + ", ".join(failed[:10]))
            else:
                df = pd.read_csv(io.StringIO(data))
                # если нет координат, но есть адреса — геокодируем их
                need_geo = ("lat" not in df.columns or "lon" not in df.columns) \
                    and "address" in df.columns
                if need_geo:
                    found, failed = load_places_by_addresses(
                        list(df["address"].astype(str)), country_bias=country_bias)
                    loaded = found
                    if failed:
                        st.warning("Не найдено адресов: " + ", ".join(failed[:10]))
                else:
                    for _, r in df.iterrows():
                        loaded.append(Place(str(r.get("name", "")), float(r["lat"]),
                                            float(r["lon"]),
                                            address=str(r.get("address", "")),
                                            category=str(r.get("category", ""))))
            if loaded:
                if f_mode_replace:
                    st.session_state.places = loaded
                else:
                    st.session_state.places.extend(loaded)
                st.success(f"Загружено: {len(loaded)}")
        except Exception as exc:
            st.error(f"Ошибка чтения файла: {exc}")

df = pd.DataFrame([{"name": p.label(), "lat": p.lat, "lon": p.lon,
                    "category": p.category} for p in st.session_state.places])
st.dataframe(df, use_container_width=True, height=180)
st.caption(f"Всего мест: {len(st.session_state.places)}")

# --------------------------------------------------------------- 2. Кластеризация
st.header("2️⃣ Кластеризация по плотности")
k1, k2, k3 = st.columns(3)
with k1:
    method = st.selectbox("Алгоритм", ["dbscan", "hdbscan"],
                          help="HDBSCAN сам подбирает плотность; без пакета hdbscan "
                               "используется DBSCAN с адаптивным eps")
with k2:
    eps_m = st.slider("Радиус окрестности eps, м", 100, 3000, 500, step=50,
                      disabled=(method == "hdbscan"))
with k3:
    min_samples = st.slider("Мин. мест в кластере", 2, 10, 3)

clusters, noise = [], list(st.session_state.places)
if True:  # пересчитываем при каждом рендере — параметры могут меняться
    clusters, noise = cluster_places(st.session_state.places,
                                     eps_m=eps_m, min_samples=min_samples,
                                     method=method)
    labels = ([c.cid for c in clusters for _ in c.members] + [-1] * len(noise))
    ordered = [p for c in clusters for p in c.members] + noise
    sil = silhouette_check(ordered, labels) if len(ordered) == len(st.session_state.places) else None
    st.write(f"**Кластеров: {len(clusters)}, вне кластеров (шум): {len(noise)}**"
             + (f" · силуэт: {sil:.2f}" if sil is not None else ""))
    for c in clusters:
        st.markdown(f"- 🔴 **Кластер {c.cid}** — {c.size} мест, радиус ≈ {c.radius_m():.0f} м, "
                    f"центр: `{c.center[0]:.5f}, {c.center[1]:.5f}`")

# --------------------------------------------------------------- 3. Маршрут
st.header("3️⃣ Оптимальный маршрут")
m1, m2 = st.columns(2)
with m1:
    mode = st.radio("Уровень оптимизации",
                    ["clusters", "places"],
                    format_func=lambda x: {"clusters": "По КЛАСТЕРАМ (сначала кластеры, внутри — места)",
                                           "places": "По всем МЕСТАМ сразу"}[x])
with m2:
    depot_s = st.text_input("Старт/финиш (лат, лон) — необязательно",
                            placeholder="59.9385, 30.3370")
include_noise = st.checkbox("Включить места вне кластеров", value=True)

route = st.session_state.get("route")
if st.button("🛣 Построить маршрут", type="primary"):
    depot = parse_coords(depot_s) if depot_s else None
    route = plan_route(clusters, noise=noise, depot=depot,
                       mode=mode, include_noise=include_noise)
    st.session_state.route = route
if route:
    for line in route.summary_lines():
        st.markdown(line)
    st.dataframe(pd.DataFrame([{
        "№": i, "Место": s.name, "Кластер": s.cluster_id or "—",
        "Тип": s.kind, "Широта": round(s.lat, 6), "Долгота": round(s.lon, 6)}
        for i, s in enumerate(route.stops, 1)]), use_container_width=True, height=220)

# --------------------------------------------------------------- 4. Карта
st.header("4️⃣ Карта")
if clusters:
    fmap = render_map(st.session_state.places, clusters=clusters, noise=noise,
                      route=route, title="Кластеры и маршрут")
    st.components.v1.html(fmap._repr_html_(), height=650, scrolling=False)
else:
    st.info("Сначала выполните кластеризацию.")

st.download_button("💾 Скачать карту HTML",
                   data=fmap.get_root().render().encode("utf-8") if clusters else b"",
                   file_name="map_clusters_route.html", mime="text/html")
