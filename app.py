"""Веб-интерфейс: поиск мест, кластеризация по плотности, оптимальный маршрут.

Запуск:  streamlit run app.py
Требуется Python 3.10+, а также: folium, scikit-learn, streamlit (опционально).
"""
from __future__ import annotations

import io
import json

import pandas as pd
import streamlit as st

from clustering import cluster_places, silhouette_check
from geo_utils import parse_coords, path_length_m
from places import CATEGORY_TAGS, Place, find_places, geocode
from routing import plan_route
from visualization import render_map

st.set_page_config(page_title="Кластеры и маршруты на карте", page_icon="🗺️", layout="wide")
st.title("🗺️ Кластеризация мест на карте и оптимальный маршрут")
st.caption("Находит места (OpenStreetMap), объединяет их в кластеры по плотности "
           "(DBSCAN/HDBSCAN) и строит оптимальный маршрут — по отдельным местам или "
           "по кластерам (двухуровневая TSP-эвристика: nearest-neighbor + 2-opt + branch&bound).")

# --------------------------------------------------------------- 1. Поиск мест
st.header("1️⃣ Места")
tab_search, tab_manual, tab_file = st.tabs(["🔎 Поиск на карте", "✍️ Вручную", "📄 Из файла"])

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

with tab_manual:
    name = st.text_input("Название места", key="m_name")
    coords_s = st.text_input("Координаты «широта, долгота»", key="m_coords",
                             placeholder="59.9398, 30.3146")
    if st.button("➕ Добавить место"):
        pc = parse_coords(coords_s)
        if not pc:
            st.error("Некорректные координаты.")
        else:
            st.session_state.places.append(
                Place(name or f"Точка {len(st.session_state.places)+1}", pc[0], pc[1],
                      source="manual"))
            st.success("Добавлено.")

with tab_file:
    upl = st.file_uploader("CSV/JSON со столбцами name,lat,lon[,category,address]",
                           type=["csv", "json"])
    if upl is not None and st.button("📥 Загрузить"):
        data = upl.read().decode("utf-8-sig")
        loaded: list[Place] = []
        try:
            if upl.name.endswith(".json"):
                for r in json.loads(data):
                    loaded.append(Place(str(r.get("name", "")), float(r["lat"]), float(r["lon"]),
                                        address=str(r.get("address", "")),
                                        category=str(r.get("category", ""))))
            else:
                df = pd.read_csv(io.StringIO(data))
                for _, r in df.iterrows():
                    loaded.append(Place(str(r.get("name", "")), float(r["lat"]), float(r["lon"]),
                                        address=str(r.get("address", "")),
                                        category=str(r.get("category", ""))))
            st.session_state.places = loaded
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
