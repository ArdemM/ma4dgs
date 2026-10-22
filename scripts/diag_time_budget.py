"""Zeitliche Verteilung des Gaußschen-Budgets aus einem Checkpoint.

Für jeden Frame t wird die Kapazität K(t) = Σᵢ σᵢ(t) berechnet (Summe der
zeitlichen Opazitäten, Gl. 4), dazu pro Frame-Bin: Anzahl der Gaußschen mit μₜ
im Bin ("geboren"), mediane Dauer und Geschwindigkeit, mittlere Opazität, sowie
der Anteil der Gaußschen außerhalb der Sequenz. Wenn stats/val_step*.json
vorliegt, zusätzlich PSNR/LPIPS pro Bin.

Aufruf: python scripts/diag_time_budget.py <result_dir> [<result_dir> ...] [--bin 10] [--csv]
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch


def analyze(result_dir: Path, bin_size: int) -> dict:
    ck = sorted(glob.glob(str(result_dir / "ckpts" / "*.pt")))[-1]
    c = torch.load(ck, map_location="cpu", weights_only=False)
    sp = c["splats"] if "splats" in c else c
    mu = sp["times"].squeeze(-1).numpy().astype(np.float64)
    s = torch.exp(sp["durations"]).squeeze(-1).numpy().astype(np.float64)
    speed = sp["velocities"].norm(dim=-1).numpy()
    opa = torch.sigmoid(sp["opacities"]).numpy()

    vals = sorted(glob.glob(str(result_dir / "stats" / "val_step*.json")))
    per_frame = json.load(open(vals[-1]))["per_frame"] if vals else []
    if per_frame:
        F = max(r["frame"] for r in per_frame) + 1
    else:
        cfg = result_dir / "cfg.yml"
        F = 0
        if cfg.exists():
            for line in open(cfg):
                if line.startswith("num_frames_effective"):
                    F = int(line.split(":")[1])
        if F <= 0:
            raise SystemExit("Frame-Anzahl unbekannt (keine val_step*.json)")
    t = np.linspace(0.0, 1.0, F)
    cap = np.array([np.exp(-0.5 * ((ti - mu) / s) ** 2).sum() for ti in t])
    out = {
        "ckpt": ck,
        "N": int(len(mu)),
        "frames": int(F),
        "outside_frac": float(((mu < 0) | (mu > 1)).mean()),
        "never_visible_frac": float(((mu < -3 * s) | (mu > 1 + 3 * s)).mean()),
        "duration_median_frames": float(np.median(s) * (F - 1)),
        "capacity": cap,
        "rows": [],
    }
    fr = np.clip(np.round(mu * (F - 1)).astype(int), 0, F - 1)
    for f0 in range(0, F, bin_size):
        f1 = min(f0 + bin_size, F)
        m = (fr >= f0) & (fr < f1)
        row = {
            "bin": f"{f0}-{f1 - 1}",
            "capacity": float(cap[f0:f1].mean()),
            "born": int(m.sum()),
            "dur_frames": float(np.median(s[m]) * (F - 1)) if m.any() else float("nan"),
            "speed": float(np.median(speed[m])) if m.any() else float("nan"),
            "opa": float(opa[m].mean()) if m.any() else float("nan"),
        }
        rows = [r for r in per_frame if f0 <= r["frame"] < f1]
        if rows:
            row["psnr"] = float(np.mean([r["psnr"] for r in rows]))
            row["lpips"] = float(np.mean([r["lpips"] for r in rows]))
        out["rows"].append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dirs", nargs="+")
    ap.add_argument("--bin", type=int, default=10)
    ap.add_argument("--csv", action="store_true", help="capacity.csv und time_budget.json in stats/ ablegen")
    args = ap.parse_args()
    for d in args.result_dirs:
        d = Path(d)
        r = analyze(d, args.bin)
        print(f"== {d}  N={r['N']}  Frames={r['frames']}")
        print(
            f"   mu_t ausserhalb [0,1]: {100 * r['outside_frac']:.1f} %   nie sichtbar (>3s ausserhalb): "
            f"{100 * r['never_visible_frac']:.1f} %   Dauer-Median {r['duration_median_frames']:.2f} Frames"
        )
        has_metrics = any("psnr" in row for row in r["rows"])
        head = f"   {'Frames':>9} {'K(t)':>8} {'geboren':>8} {'Dauer':>6} {'|v|':>7} {'Opa':>6}"
        if has_metrics:
            head += f" {'PSNR':>6} {'LPIPS':>6}"
        print(head)
        for row in r["rows"]:
            line = (
                f"   {row['bin']:>9} {row['capacity']:8.0f} {row['born']:8d} "
                f"{row['dur_frames']:6.2f} {row['speed']:7.4f} {row['opa']:6.3f}"
            )
            if "psnr" in row:
                line += f" {row['psnr']:6.2f} {row['lpips']:6.3f}"
            print(line)
        if args.csv:
            np.savetxt(
                d / "stats" / "capacity.csv",
                np.c_[np.arange(r["frames"]), r["capacity"]],
                fmt="%d,%.1f", header="frame,capacity", comments="",
            )
            with open(d / "stats" / "time_budget.json", "w") as f:
                json.dump({k: v for k, v in r.items() if k != "capacity"}, f, indent=2)


if __name__ == "__main__":
    main()
