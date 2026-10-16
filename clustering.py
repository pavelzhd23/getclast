"""Кластеризация мест по плотности расположения (DBSCAN / HDBSCAN).

Алгоритмы Density-Based объединяют в один кластер места, которые находятся
«густо» относительно друг друга: цепочка точек, где каждая следующая — в пределах
eps от предыдущей, попадает в один кластер. Точки, не попавшие ни в одну плотную
область, помечаются как шум (cluster = -1) и могут обрабатываться отдельно.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from geo_utils import distance_matrix_m, spherical_centroid
from places import Place


@dataclass
class Cluster:
    """Кластер мест: идентификатор, состав, центроид, радиус."""
    cid: int
    members: list[Place] = field(default_factory=list)

    @property
    def center(self) -> tuple[float, float]:
        return spherical_centroid([p.coords for p in self.members])

    @property
    def size(self) -> int:
        return len(self.members)

    def radius_m(self) -> float:
        """Радиус — максимальное расстояние от центра до члена кластера."""
        c = self.center
        return max(_hav(c, p.coords) for p in self.members) if self.size > 1 else 0.0

    def label(self) -> str:
        names = ", ".join(p.label() for p in self.members[:5])
        if self.size > 5:
            names += f" … (+{self.size - 5})"
        return f"Кластер {self.cid} ({self.size}): {names}"


def _hav(a, b) -> float:
    from geo_utils import haversine_m
    return haversine_m(a[0], a[1], b[0], b[1])


def cluster_places(places: list[Place], eps_m: float = 400.0, min_samples: int = 3,
                   method: str = "dbscan") -> tuple[list[Cluster], list[Place]]:
    """Разбивает места на кластеры по плотности.

    Параметры:
      eps_m       — радиус окрестности в метрах (макс. расстояние «соседства»);
      min_samples — минимальное число мест в окрестности для «ядровой» точки;
      method      — 'dbscan' или 'hdbscan'.

    Возвращает (clusters, noise): список кластеров и список шумовых точек.
    """
    if len(places) < min_samples:
        return [], list(places)

    D = distance_matrix_m([p.coords for p in places])

    labels: np.ndarray
    if method == "hdbscan":
        try:
            from hdbscan import HDBSCAN
            model = HDBSCAN(min_cluster_size=max(min_samples, 2),
                            metric="precomputed",
                            allow_single_cluster=True)
            labels = model.fit_predict(D)
        except ImportError:
            # Fallback: DBSCAN с адаптивным eps по k-distance графику
            labels = _dbscan_adaptive_eps(D, min_samples)
    else:
        from sklearn.cluster import DBSCAN
        model = DBSCAN(eps=eps_m, min_samples=min_samples, metric="precomputed")
        labels = model.fit_predict(D)

    clusters: list[Cluster] = []
    noise: list[Place] = []
    uniq = sorted(set(int(l) for l in labels) - {-1})
    for new_id, old_id in enumerate(uniq, start=1):
        members = [places[i] for i in range(len(places)) if labels[i] == old_id]
        clusters.append(Cluster(cid=new_id, members=members))
    for i, l in enumerate(labels):
        if int(l) == -1:
            noise.append(places[i])

    # «шум» из двух и более взаимно близких точек (eps < d <= 2*eps):
    # DBSCAN их не склеил, но по плотности это явно группа — объединяем.
    extra: list[Cluster] = []
    remaining: list[Place] = []
    used = [False] * len(noise)
    for i, p in enumerate(noise):
        if used[i]:
            continue
        group = [p]
        used[i] = True
        for j in range(i + 1, len(noise)):
            if not used[j] and _hav(p.coords, noise[j].coords) <= 2 * eps_m:
                group.append(noise[j])
                used[j] = True
        if len(group) >= 2:
            extra.append(Cluster(cid=0, members=group))
        else:
            remaining.append(p)
    if extra:
        next_cid = max([c.cid for c in clusters], default=0) + 1
        for cl in extra:
            cl.cid = next_cid
            next_cid += 1
        clusters.extend(extra)
        noise = remaining

    return clusters, noise


def _dbscan_adaptive_eps(D: np.ndarray, min_samples: int) -> np.ndarray:
    """DBSCAN с eps, выбранным по k-distance графику (95-й перцентиль)."""
    from sklearn.cluster import DBSCAN
    k_dist = np.sort(D, axis=1)[:, min_samples - 1]
    eps = float(np.percentile(k_dist[k_dist > 0], 95)) if np.any(k_dist > 0) else 400.0
    return DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit_predict(D)


def silhouette_check(places: list[Place], labels: list[int]) -> float | None:
    """Оценка качества кластеризации (silhouette на метрике haversine). None если неприменимо."""
    from sklearn.metrics import silhouette_score
    if len(set(labels) - {-1}) < 2:
        return None
    idx = [i for i, l in enumerate(labels) if l != -1]
    if len(idx) < 3:
        return None
    D = distance_matrix_m([places[i].coords for i in idx])
    lab = [labels[i] for i in idx]
    return float(silhouette_score(D, lab, metric="precomputed"))
