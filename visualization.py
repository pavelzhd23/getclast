"""Визуализация мест, кластеров и маршрута на интерактивной карте (folium)."""
from __future__ import annotations

import folium
from folium import plugins
from branca.element import MacroElement, Template

from clustering import Cluster
from places import Place
from routing import RouteResult

PALETTE = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
           "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62"]


def cluster_color(cid: int | None) -> str:
    if not cid or cid < 1:
        return "#808080"
    return PALETTE[(cid - 1) % len(PALETTE)]


class _HtmlControl(MacroElement):
    """Произвольный HTML-блок в углу карты средствами Leaflet Control API."""
    _template = Template("""
        {% macro script(this, kwargs) %}
            var {{ this.get_name() }} = L.control({ position: 'topright' });
            {{ this.get_name() }}.onAdd = function (_) {
                var div = L.DomUtil.create('div', 'leaflet-control');
                div.innerHTML = `{{ this.html | safe }}`;
                return div;
            };
            {{ this.get_name() }}.addTo({{ this._parent.get_name() }});
        {% endmacro %}
        """)

    def __init__(self, html: str):
        super().__init__()
        self._name = "legend_ctl_" + str(abs(hash(html)) % 10_000)
        self.html = html.replace("`", "'").replace("\n", " ")


def render_map(places: list[Place],
               clusters: list[Cluster] | None = None,
               route: RouteResult | None = None,
               noise: list[Place] | None = None,
               center: tuple[float, float] | None = None,
               zoom: int = 13,
               title: str = "Кластеризация и маршрут") -> folium.Map:
    """Собирает карту: маркеры мест (окрашены по кластерам), овалы охвата, линия маршрута."""
    all_pts = places or []
    if center is None:
        if all_pts:
            lats = [p.lat for p in all_pts]
            lons = [p.lon for p in all_pts]
            center = ((min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2)
        else:
            center = (55.75, 37.62)

    m = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap")

    cmap: dict[int, int] = {}
    if clusters:
        for c in clusters:
            for p in c.members:
                cmap[id(p)] = c.cid

    # --- маркеры мест
    fg_places = folium.FeatureGroup(name="Места", show=True)
    for p in all_pts:
        cid = cmap.get(id(p))
        color = cluster_color(cid) if cid else "#808080"
        popup = (f"<b>{p.label()}</b><br>{p.address}<br>"
                 f"категория: {p.category or '—'}<br>кластер: {cid if cid else '—'}")
        folium.Marker(
            location=p.coords,
            popup=folium.Popup(popup, max_width=350),
            tooltip=f"{p.label()}" + (f" [К{cid}]" if cid else ""),
            icon=folium.Icon(color=_folium_color_name(color), prefix="fa", icon="map-marker"),
        ).add_to(fg_places)
    fg_places.add_to(m)

    # --- овалы охвата кластеров + центры
    if clusters:
        fg_cl = folium.FeatureGroup(name="Кластеры (охват)", show=True)
        for c in clusters:
            col = cluster_color(c.cid)
            folium.Circle(location=c.center, radius=max(c.radius_m() * 2.2, 120),
                          color=col, weight=2, fill=True, fill_opacity=0.12,
                          tooltip=c.label()).add_to(fg_cl)
            folium.Marker(location=c.center,
                          icon=folium.DivIcon(
                              html=(f'<div style="background:{col};color:#fff;border-radius:6px;'
                                    f'width:24px;height:24px;text-align:center;line-height:24px;'
                                    f'font-size:13px;font-weight:bold;border:2px solid #fff;'
                                    f'box-shadow:0 1px 3px rgba(0,0,0,.5);">{c.cid}</div>'),
                              icon_size=(24, 24), icon_anchor=(12, 12)),
                          tooltip=f"Центр кластера {c.cid}").add_to(fg_cl)
        fg_cl.add_to(m)

    # --- маршрут
    if route and route.stops:
        fg_rt = folium.FeatureGroup(name="Маршрут", show=True)
        coords = [s.coords for s in route.stops]
        folium.PolyLine(coords, color="#2222aa", weight=4, opacity=0.85,
                        tooltip=f"Маршрут ≈ {route.total_km:.2f} км").add_to(fg_rt)
        n = len(route.stops)
        for i, s in enumerate(route.stops, start=1):
            if s.kind == "depot":
                lbl, col = ("⚑" if i == n else "A"), "#000000"
            else:
                lbl, col = str(i), cluster_color(s.cluster_id)
            folium.Marker(
                location=s.coords,
                tooltip=f"{i}. {s.name}",
                icon=folium.DivIcon(
                    html=(f'<div style="background:{col};color:#fff;border-radius:50%;'
                          f'width:22px;height:22px;text-align:center;line-height:22px;'
                          f'font-size:12px;font-weight:bold;border:2px solid #fff;'
                          f'box-shadow:0 1px 3px rgba(0,0,0,.5);">{lbl}</div>'),
                    icon_size=(22, 22), icon_anchor=(11, 11)),
            ).add_to(fg_rt)
        fg_rt.add_to(m)

    # --- легенда в углу карты
    items = [f"<h4 style='margin:2px 0'>{title}</h4><ul style='margin:4px;padding-left:18px'>"]
    if clusters:
        for c in clusters:
            items.append(f'<li><span style="color:{cluster_color(c.cid)}">●</span> '
                         f'Кластер {c.cid}: {c.size} мест (радиус ~{c.radius_m():.0f} м)</li>')
    if noise:
        items.append(f'<li><span style="color:#808080">●</span> Вне кластеров (шум): {len(noise)}</li>')
    if route:
        items.append(f"<li>🛣 Маршрут: {route.total_km:.2f} км, остановок: {len(route.stops)}</li>")
    items.append("</ul>")
    legend_html = ('<div style="padding:6px;background:white;border-radius:6px;'
                   'box-shadow:0 1px 4px rgba(0,0,0,.35);max-width:340px;font-size:12px">'
                   + "".join(items) + "</div>")
    _HtmlControl(legend_html).add_to(m)

    plugins.Fullscreen(position="topleft").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def _folium_color_name(hex_color: str) -> str:
    """hex -> ближайший именованный цвет иконки folium."""
    table = {
        "#e41a1c": "red", "#377eb8": "blue", "#4daf4a": "green", "#984ea3": "purple",
        "#ff7f00": "orange", "#a65628": "brown", "#f781bf": "pink", "#999999": "gray",
        "#66c2a5": "lightgreen", "#fc8d62": "orange", "#808080": "gray", "#000000": "black",
    }
    return table.get(hex_color, "blue")
