"""Geschwindigkeit aus einem 3D-Skelett (eigene Erweiterung, nicht Teil von FreeTimeGS).

4DAnyone erzeugt die Ansichten aus einem 3D-Skelett, das GVHMR aus dem
Eingabevideo schätzt (70 Goliath-Punkte pro Frame, conditioning/keypoints_3d.npy),
und die generierten Videos folgen diesem Skelett. Damit ist für jeden Punkt in
Körpernähe bekannt, wie er sich bewegt, auch dort, wo das Bild es nicht zeigt:
Bewegt sich einfarbige Kleidung entlang ihrer eigenen Oberfläche, ändert sich
kein Pixel, und der Bildfehler liefert keinen Gradienten für v.

Geschwindigkeit eines Punkts x zum Zeitpunkt t:
  1. Für jeden Knochen (a, b): nächster Punkt auf der Strecke a–b mit Parameter
     u ∈ [0, 1], Versatz r = x − nächster Punkt, Abstand d = |r|.
  2. Geschwindigkeit dieses Knochens an x:
         (1 − u)·v_a + u·v_b + ω × r,   ω = (b − a) × (v_b − v_a) / |b − a|²
     (Drehung des Knochens; die Drehung um die eigene Achse ist aus zwei Punkten
     nicht bestimmbar und fehlt).
  3. Gewichtetes Mittel über alle Knochen, Gewichte softmax(−d² / 2σ²).
Konfidenz c = exp(−d_min² / 2τ²): Seil, Haare und Reste weit weg vom Skelett
bekommen kaum Gewicht.

v_a sind zentrale Differenzen der Punkte über die Frames; zwischen Frames wird
linear interpoliert.
"""

from typing import Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

# Knochen als Paare von Goliath-Punktnamen. Grundlage sind die Verbindungen, die
# 4DAnyone für sein Goliath40-Skelett zeichnet (fdanyone/skeleton/keypoints.py,
# nach Sapiens2), ohne Fingerglieder; dazu Hals–Nase und die Augen-Querverbindung.
BONES: Tuple[Tuple[str, str], ...] = (
    # Beine und Füße
    ("left-hip", "left-knee"), ("left-knee", "left-ankle"),
    ("right-hip", "right-knee"), ("right-knee", "right-ankle"),
    ("left-ankle", "left-big-toe-tip"), ("left-ankle", "left-small-toe-tip"), ("left-ankle", "left-heel"),
    ("right-ankle", "right-big-toe-tip"), ("right-ankle", "right-small-toe-tip"), ("right-ankle", "right-heel"),
    # Rumpf mit Diagonalen
    ("left-hip", "right-hip"), ("left-shoulder", "right-shoulder"),
    ("left-shoulder", "left-hip"), ("right-shoulder", "right-hip"),
    ("left-shoulder", "right-hip"), ("right-shoulder", "left-hip"),
    # Arme, Hand bis zu den Fingergrundgelenken
    ("left-shoulder", "left-elbow"), ("left-elbow", "left-wrist"),
    ("right-shoulder", "right-elbow"), ("right-elbow", "right-wrist"),
    *((f"{side}-wrist", f"{side}-{finger}-third-joint")
      for side in ("left", "right") for finger in ("thumb", "index", "middle", "ring", "pinky")),
    # Kopf und Hals
    ("nose", "left-eye"), ("nose", "right-eye"), ("left-eye", "right-eye"),
    ("left-eye", "left-ear"), ("right-eye", "right-ear"), ("left-ear", "right-ear"),
    ("left-ear", "left-shoulder"), ("right-ear", "right-shoulder"), ("neck", "nose"),
)


class SkeletonPrior:
    def __init__(
        self,
        keypoints: np.ndarray,
        names: Sequence[str],
        times: np.ndarray,
        meter: float,
        sigma: float = 0.05,
        tau: float = 0.1,
        device: str = "cuda",
    ):
        """keypoints [F, K, 3] und times [F] (normierte Zeit, gleichmäßig) im
        Koordinatensystem des Trainings; meter = Länge von 1 m darin;
        sigma (Zuordnung zu Knochen) und tau (Konfidenz) in Metern."""
        index = {n: i for i, n in enumerate(names)}
        self.a = torch.tensor([index[a] for a, _ in BONES], device=device)
        self.b = torch.tensor([index[b] for _, b in BONES], device=device)
        P = torch.tensor(keypoints, dtype=torch.float32, device=device)
        T = torch.tensor(times, dtype=torch.float32, device=device)
        assert len(P) >= 2 and len(P) == len(T), (P.shape, T.shape)
        V = torch.empty_like(P)
        V[1:-1] = (P[2:] - P[:-2]) / (T[2:] - T[:-2])[:, None, None]
        V[0] = (P[1] - P[0]) / (T[1] - T[0])
        V[-1] = (P[-1] - P[-2]) / (T[-1] - T[-2])
        self.P, self.V, self.T = P, V, T
        self.inv_2sigma2 = 1.0 / (2.0 * (sigma * meter) ** 2)
        self.inv_2tau2 = 1.0 / (2.0 * (tau * meter) ** 2)

    def state(self, t: float) -> Tuple[Tensor, Tensor]:
        """Punkte und Geschwindigkeiten [K, 3] zum Zeitpunkt t (linear zwischen Frames)."""
        T = self.T
        s = (min(max(t, float(T[0])), float(T[-1])) - float(T[0])) / float(T[-1] - T[0]) * (len(T) - 1)
        i = min(int(s), len(T) - 2)
        w = s - i
        return (1 - w) * self.P[i] + w * self.P[i + 1], (1 - w) * self.V[i] + w * self.V[i + 1]

    @torch.no_grad()
    def query(self, x: Tensor, t: float, chunk: int = 65536) -> Tuple[Tensor, Tensor]:
        """x [M, 3] -> Geschwindigkeit [M, 3] (Koordinaten pro normierter Zeit), Konfidenz [M]."""
        P, V = self.state(t)
        pa, pb, va, vb = P[self.a], P[self.b], V[self.a], V[self.b]  # [B, 3]
        d = pb - pa
        len2 = (d * d).sum(-1).clamp_min(1e-12)  # [B]
        omega = torch.cross(d, vb - va, dim=-1) / len2[:, None]  # [B, 3]
        vels, confs = [], []
        for xs in torch.split(x.detach().float(), chunk):
            rel = xs[:, None] - pa  # [m, B, 3]
            u = ((rel * d).sum(-1) / len2).clamp(0.0, 1.0)  # [m, B]
            r = rel - u[..., None] * d
            dist2 = (r * r).sum(-1)  # [m, B]
            vel = va + u[..., None] * (vb - va) + torch.cross(omega.expand_as(r), r, dim=-1)
            w = torch.softmax(-dist2 * self.inv_2sigma2, dim=1)
            vels.append((w[..., None] * vel).sum(1))
            confs.append(torch.exp(-dist2.min(1).values * self.inv_2tau2))
        return torch.cat(vels), torch.cat(confs)
