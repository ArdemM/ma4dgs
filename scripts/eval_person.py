"""Fehler nur im Personenbereich der Testkameras (4DAnyone-Szenen mit masks/).

train.py misst PSNR über das ganze Bild mit schwarzem Hintergrund. Die Person bedeckt
nur 12–15 % der Pixel, deshalb liegen diese Werte etwa 8–9 dB zu hoch. Dieses Skript
rendert die Testkameras aus ckpts/ckpt_final.pt neu und misst
  - PSNR und mittleren absoluten Fehler (Graustufen 0–255) im Bereich R = Maske der
    Referenz (> 0,5), um --dilate Pixel erweitert (für alle Läufe derselbe Bereich),
  - den Anteil des quadratischen Fehlers, der außerhalb von R liegt (Prüfung der Annahme,
    dass außerhalb der Person kaum Fehler entsteht),
  - zum Vergleich den PSNR über das ganze Bild (wie train.py),
  - die Deckung (Summe der gerenderten Opazität) bei t = Frame + 0,5 relativ zum Mittel der
    beiden Nachbarframes: Maß dafür, ob das Modell zwischen den Frames stetig bleibt
    (keine Referenzbilder dafür vorhanden, daher nur Deckung, keine Bildqualität).
Referenz und Hintergrund wie in train.py (Bild·Maske, Hintergrund schwarz).

Läufe auf einem Zeitfenster werden nur auf ihren Frames ausgewertet. Statische
Einzelbild-Läufe (Datensatz mit einem Frame) zählen als der Frame, auf den ihr Bild
verlinkt (48Views_static_fNN), sonst als --static-frame.

Aufruf: python scripts/eval_person.py <result_dir> [...] [--frames 50 70] [--static-frame 60]
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train import Config, Runner  # noqa: E402

PHASES = {"ruhig 0-39": (0, 39), "Sprung 50-70": (50, 70), "f60": (60, 60)}


def load_runner(result_dir: Path, f0: int, f1: int, static_frame: int):
    """Runner nur mit den Frames f0..f1 (bzw. dem Einzelbild) und den Splats des Checkpoints."""
    raw = yaml.safe_load(open(result_dir / "cfg.yml"))
    fields = Config.__dataclass_fields__
    cfg = Config(**{k: v for k, v in raw.items() if k in fields})
    first = Path(cfg.data_dir).expanduser() / "images" / cfg.test_cams[0] / "00000.png"
    n_avail = len(list(first.parent.glob("*.png")))
    if n_avail == 1:  # statischer Einzelbild-Lauf
        # Diagnose-Datensätze (48Views_static_fNN) verlinken das Bild des Frames der Sequenz
        offset = int(first.resolve().stem) if first.is_symlink() else static_frame
        static_frame = offset
        if not f0 <= static_frame <= f1:
            return None, None
    else:
        start = cfg.frame_start or 0
        end = start + (cfg.num_frames if cfg.num_frames > 0 else n_avail - start) - 1
        lo, hi = max(f0, start), min(f1, end)
        if lo > hi:
            return None, None
        if cfg.frame_start is None and cfg.num_frames > 0:
            raise ValueError(f"{result_dir}: --num-frames ohne --frame-start, Zeitnormierung nicht wiederherstellbar")
        cfg.frame_start, cfg.num_frames = lo, hi - lo + 1
        offset = 0
    cfg.result_dir = tempfile.mkdtemp(prefix="eval_person_")
    cfg.init_velocity = "zero"  # Init wird ohnehin durch den Checkpoint ersetzt
    cfg.lambda_skel = 0.0
    cfg.load_workers = 8
    runner = Runner(cfg)
    ckpt = torch.load(result_dir / "ckpts" / "ckpt_final.pt", map_location="cuda", weights_only=False)
    for k in runner.splats.keys():
        runner.splats[k].data = ckpt["splats"][k]
    return runner, offset


@torch.no_grad()
def evaluate(runner: Runner, offset: int, dilate: int):
    data = runner.data
    kernel = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
    bg = torch.zeros((1, 3), device="cuda")
    rows = []
    for cam in data.test_cams:
        alpha_sum = []  # gerenderte Deckung (Summe der Opazität) pro Frame
        for f in range(data.num_frames):
            pixels, m = runner.target(cam, f, bg)
            colors, alphas, _, _ = runner.render(cam, float(data.times[f]), runner.cfg.sh_degree, bg=bg)
            alpha_sum.append(alphas.sum().item())
            colors = colors.clamp(0, 1)
            region = cv2.dilate((m[0, ..., 0].cpu().numpy() > 0.5).astype(np.uint8), kernel) > 0
            region = torch.from_numpy(region).cuda()
            se = ((colors - pixels) ** 2).mean(-1)[0]  # [H, W]
            ae = (colors - pixels).abs().mean(-1)[0] * 255.0
            mse_r = se[region].mean().item()
            rows.append({
                "cam": data.cam_names[cam],
                "frame": offset + data.frame_start + f,
                "psnr_person": -10 * np.log10(max(mse_r, 1e-12)),
                "mae_person": ae[region].mean().item(),
                "psnr_full": -10 * np.log10(max(se.mean().item(), 1e-12)),
                "outside_err_frac": se[~region].sum().item() / max(se.sum().item(), 1e-12),
                "region_frac": region.float().mean().item(),
            })
        # Zeitliche Stetigkeit: Deckung bei t = Frame + 0,5 relativ zum Mittel der beiden Frames.
        # Ein stetiges 4D-Modell liegt nahe 1; ein Pro-Frame-Modell wird dazwischen durchsichtig.
        n = data.num_frames
        cam_rows = rows[-n:]
        for f in range(n - 1):
            t_mid = 0.5 * float(data.times[f] + data.times[f + 1])
            _, alphas, _, _ = runner.render(cam, t_mid, runner.cfg.sh_degree, bg=bg)
            cam_rows[f]["coverage_mid"] = alphas.sum().item() / max(0.5 * (alpha_sum[f] + alpha_sum[f + 1]), 1e-6)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dirs", nargs="+")
    ap.add_argument("--frames", type=int, nargs=2, default=[0, 70], help="erster und letzter Frame (Sequenz)")
    ap.add_argument("--static-frame", type=int, default=60)
    ap.add_argument("--dilate", type=int, default=5, help="Erweiterung der Maske in Pixeln")
    args = ap.parse_args()
    print("Lauf | Phase | PSNR Person | MAE Person (0–255) | PSNR ganzes Bild | Fehleranteil außerhalb | Bildanteil R"
          " | Deckung bei Frame + 0,5")
    for d in map(lambda p: Path(p).expanduser(), args.result_dirs):
        runner, offset = load_runner(d, *args.frames, args.static_frame)
        if runner is None:
            print(f"{d.name}: keine Frames im Bereich")
            continue
        rows = evaluate(runner, offset, args.dilate)
        (d / "stats" / "person_metrics.json").write_text(json.dumps({"dilate": args.dilate, "per_frame": rows}, indent=1))
        for ph, (f0, f1) in PHASES.items():
            sel = [r for r in rows if f0 <= r["frame"] <= f1]
            if not sel:
                continue
            mean = {k: float(np.mean([r[k] for r in sel])) for k in
                    ["psnr_person", "mae_person", "psnr_full", "outside_err_frac", "region_frac"]}
            mids = [r["coverage_mid"] for r in sel if "coverage_mid" in r]
            cov = f"{np.mean(mids):.3f} (min {np.min(mids):.3f})" if mids else "–"
            print(f"{d.name} | {ph} | {mean['psnr_person']:.2f} | {mean['mae_person']:.2f} | {mean['psnr_full']:.2f} | "
                  f"{mean['outside_err_frac'] * 100:.1f}% | {mean['region_frac'] * 100:.1f}% | {cov}", flush=True)
        del runner
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
