"""Gelernte Geschwindigkeiten gegen das 3D-Skelett und Bildqualität pro Bewegungsphase.

Für jeden Ergebnisordner (train.py mit 4DAnyone-Szene, skeleton/ im Datensatz):
  - PSNR/LPIPS der Testkameras gemittelt über Frame-Bereiche (stats/val_step*.json),
    Standard: ruhig 0–39, Sprung 50–70, alle;
  - für einzelne Frames aus ckpts/ckpt_final.pt: Gaußsche mit σ(t) > 0,5 und
    Skelett-Konfidenz > 0,5, davon Median |v|, Median der Skelett-Geschwindigkeit
    an ihrer Position, Median |v − v_skel| (alles in cm pro Frame), Median der
    Dauer s (Frames) und Anzahl.

Aufruf: python scripts/diag_skeleton.py <result_dir> [<result_dir> ...] [--frames 10 60]
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ma4dgs.data import MultiViewVideo  # noqa: E402
from ma4dgs.model import temporal_opacity  # noqa: E402
from ma4dgs.skeleton import SkeletonPrior  # noqa: E402


class SceneGeometry(MultiViewVideo):
    """Nur Kameras, Normalisierung und Skelett, ohne Bilder zu laden."""

    def _load_images(self, workers, kind="images"):
        return None


def phase_metrics(result_dir: Path, phases):
    vals = sorted(glob.glob(str(result_dir / "stats" / "val_step*.json")))
    if not vals:
        return {}
    rows = json.load(open(vals[-1]))["per_frame"]
    out = {}
    for name, (f0, f1) in phases.items():
        sel = [r for r in rows if f0 <= r["frame"] <= f1]
        out[name] = (np.mean([r["psnr"] for r in sel]), np.mean([r["lpips"] for r in sel]))
    return out


@torch.no_grad()
def velocity_stats(result_dir: Path, frames, scenes: dict):
    cfg = yaml.safe_load(open(result_dir / "cfg.yml"))
    key = cfg["data_dir"]
    if key not in scenes:
        data = SceneGeometry(key, test_cams=[], num_frames=cfg["num_frames"])
        prior = SkeletonPrior(*data.skeleton(), data.times, data.meter,
                              sigma=cfg.get("skel_sigma", 0.05), tau=cfg.get("skel_tau", 0.1))
        scenes[key] = (data, prior)
    data, prior = scenes[key]
    F = data.num_frames
    to_cm = 100.0 / ((F - 1) * data.meter)
    sp = torch.load(result_dir / "ckpts" / "ckpt_final.pt", map_location="cuda", weights_only=False)["splats"]
    out = {}
    for f in frames:
        t = float(data.times[f])
        active = temporal_opacity(sp, t) > 0.5
        v = sp["velocities"][active]
        x = sp["means"][active] + v * (t - sp["times"][active])
        v_skel, conf = prior.query(x, t)
        near = conf > 0.5
        v, v_skel = v[near], v_skel[near]
        dur = torch.exp(sp["durations"][active][near]).squeeze(-1) * (F - 1)
        out[f] = {
            "n": int(near.sum()),
            "v": float(v.norm(dim=-1).median() * to_cm),
            "v_skel": float(v_skel.norm(dim=-1).median() * to_cm),
            "diff": float((v - v_skel).norm(dim=-1).median() * to_cm),
            "dur": float(dur.median()),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dirs", nargs="+")
    ap.add_argument("--frames", type=int, nargs="+", default=[10, 60])
    args = ap.parse_args()
    phases = {"ruhig 0-39": (0, 39), "Sprung 50-70": (50, 70), "alle": (0, 10**6)}
    scenes = {}
    for d in map(Path, args.result_dirs):
        d = d.expanduser()
        print(f"== {d.name}")
        pm = phase_metrics(d, phases)
        if pm:
            print("   " + " | ".join(f"{k}: {p:.2f} dB / {l:.3f}" for k, (p, l) in pm.items()))
        for f, s in velocity_stats(d, args.frames, scenes).items():
            print(f"   f{f:3d}: {s['n']:6d} aktiv | |v| {s['v']:.2f} cm | Skelett {s['v_skel']:.2f} cm | "
                  f"|v-v_skel| {s['diff']:.2f} cm | Dauer {s['dur']:.2f} Frames")


if __name__ == "__main__":
    main()
