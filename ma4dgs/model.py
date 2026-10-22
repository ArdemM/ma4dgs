"""FreeTimeGS-Repräsentation (Wang et al., CVPR 2025, Abschnitt 3.1).

Jede Gaußsche hat zusätzlich zu den 3DGS-Parametern eine Zeit μₜ, eine Dauer s
und eine Geschwindigkeit v. Für einen Zeitpunkt t gilt

    μₓ(t) = μₓ + v·(t − μₜ)                 (Gl. 1)
    σ(t)  = exp(−½·((t − μₜ)/s)²)           (Gl. 4)
    Opazität(t) = σ · σ(t)                  (Gl. 3)

Das Ergebnis ist eine gewöhnliche 3D-Gaußsche Szene, die an
gsplat.rasterization übergeben wird. Gradienten auf μₜ, s und v laufen über
PyTorch-Autograd, ein eigener CUDA-Kernel ist nicht nötig.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from .data import GSPLAT_EXAMPLES  # noqa: F401  (setzt sys.path für utils)
from utils import knn, rgb_to_sh  # gsplat/examples/utils.py


def knn_velocities(
    frames: List[Tuple[np.ndarray, np.ndarray, float]], k: int = 3, align_centroid: bool = False
) -> List[np.ndarray]:
    """Geschwindigkeit pro Punkt aus k-NN-Zuordnung zum nächsten Keyframe (Abschnitt 3.2).

    v = (Mittel der k nächsten Nachbarn im nächsten Keyframe − p) / Δt.
    Der letzte Keyframe nimmt die Rückwärtsdifferenz zum vorherigen.
    Rückgabe: eine Liste [M, 3] pro Keyframe (Weltkoordinaten pro Zeiteinheit).

    align_centroid: Vor der Zuordnung die Punktwolke um die Schwerpunkt-
    verschiebung zum nächsten Keyframe versetzen und diese zur Geschwindigkeit
    addieren. Grund: Bei Bewegung entlang der eigenen Oberfläche (Körper
    springt nach oben, Rumpf und Beine sind senkrechte Flächen) findet die
    Nächste-Nachbar-Zuordnung den Punkt *auf derselben Fläche* und
    unterschätzt die Bewegung stark (Seilspringen: Schwerpunkt 4 cm/Frame,
    kNN-Fluss 0,3 cm/Frame). Die globale Verschiebung wird so vorab erfasst.
    """
    from sklearn.neighbors import NearestNeighbors

    if len(frames) < 2:
        return [np.zeros_like(pts) for pts, _, _ in frames]
    vels = []
    for i, (pts, _, t) in enumerate(frames):
        j = i + 1 if i + 1 < len(frames) else i - 1
        other, _, t_other = frames[j]
        shift = (other.mean(axis=0) - pts.mean(axis=0)) if align_centroid else np.zeros(3)
        nn = NearestNeighbors(n_neighbors=k).fit(other)
        _, idx = nn.kneighbors(pts + shift)
        target = other[idx].mean(axis=1)  # [M, 3]
        vels.append(((target - pts) / (t_other - t)).astype(np.float32))
    return vels


def init_from_frames(
    frames: List[Tuple[np.ndarray, np.ndarray, float]],
    init_duration: float,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    sh_degree: int = 3,
    velocities: Optional[List[np.ndarray]] = None,
) -> Dict[str, Tensor]:
    """Gaußsche aus Punktwolken pro Frame, Zeit = Zeit des Frames.

    frames: Liste von (Punkte [M, 3], Farben [M, 3] in [0, 1], normierte Zeit t).
    velocities: optional eine Geschwindigkeit [M, 3] pro Frame (knn_velocities),
    sonst v = 0.
    Die Skalierung kommt aus dem Abstand zu den 3 nächsten Nachbarn *innerhalb
    desselben Frames* (statische Bereiche wären sonst über alle Frames hinweg
    mehrfach vorhanden und die Abstände künstlich klein).
    """
    means, rgbs, times, scales = [], [], [], []
    for pts, rgb, t in frames:
        p = torch.from_numpy(pts).float()
        dist2 = (knn(p, 4)[:, 1:] ** 2).mean(dim=-1)
        means.append(p)
        rgbs.append(torch.from_numpy(rgb).float())
        times.append(torch.full((len(p), 1), float(t)))
        scales.append(torch.log(torch.sqrt(dist2) * init_scale).unsqueeze(-1).repeat(1, 3))
    means = torch.cat(means)
    N = len(means)

    sh = torch.zeros((N, (sh_degree + 1) ** 2, 3))
    sh[:, 0, :] = rgb_to_sh(torch.cat(rgbs))
    return {
        "means": means,
        "scales": torch.cat(scales),
        "quats": torch.rand((N, 4)),
        "opacities": torch.logit(torch.full((N,), init_opacity)),
        "sh0": sh[:, :1, :],
        "shN": sh[:, 1:, :],
        "times": torch.cat(times),  # μₜ [N, 1]
        "durations": torch.full((N, 1), float(np.log(init_duration))),  # log s [N, 1]
        # v [N, 3], Weltkoordinaten pro Zeiteinheit
        "velocities": (
            torch.from_numpy(np.concatenate(velocities)).float()
            if velocities is not None
            else torch.zeros((N, 3))
        ),
    }


def temporal_opacity(splats: Dict[str, Tensor], t: float) -> Tensor:
    """σ(t) aus Gl. 4, Form [N]."""
    s = torch.exp(splats["durations"]).squeeze(-1)
    dt = t - splats["times"].squeeze(-1)
    return torch.exp(-0.5 * (dt / s) ** 2)


def gaussians_at(
    splats: Dict[str, Tensor], t: float, cull_threshold: float = 1e-3
) -> Tuple[Dict[str, Tensor], Tensor]:
    """3D-Gaußsche zum Zeitpunkt t.

    Gaußsche mit σ(t) < cull_threshold tragen praktisch nichts bei und werden
    vor der Rasterisierung entfernt (zeitliches Culling).
    Rückgabe: Parameter für gsplat.rasterization und die Maske [N] (bool).
    """
    temp = temporal_opacity(splats, t)
    mask = temp > cull_threshold
    dt = t - splats["times"][mask]  # [M, 1]
    return {
        "means": splats["means"][mask] + splats["velocities"][mask] * dt,
        "quats": splats["quats"][mask],
        "scales": torch.exp(splats["scales"][mask]),
        "opacities": torch.sigmoid(splats["opacities"][mask]) * temp[mask],
        "colors": torch.cat([splats["sh0"][mask], splats["shN"][mask]], 1),
    }, mask
