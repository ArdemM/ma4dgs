"""Statische Obergrenze: Liegt die Unschärfe an den Daten oder am eigenen Trainer?

Ein Frame einer 4DAnyone-Szene wird als gewöhnliche COLMAP-Szene mit schwarz hinterlegten
Bildern (Bild · Maske) geschrieben. So kann gsplat examples/simple_trainer.py sie unverändert
trainieren (Referenz-3DGS mit Densification bzw. MCMC). `eval` misst danach pro Kamera den PSNR
über das ganze Bild und im Personenbereich (wie eval_person.py: Maske > 0,5, um --dilate Pixel
erweitert), getrennt nach Trainings- und Testkameras, für
  - den gsplat-Checkpoint und
  - optional einen statischen Lauf des eigenen Trainers (z. B. results/4danyone/48Views_static_f10).
Trifft auch die Referenz die Trainingskameras nicht deutlich besser, liegt die Grenze bei den
Daten (Ansichten oder Posen passen nicht zusammen), sonst beim eigenen statischen Aufbau.

Aufruf:
    python scripts/diag_static_ref.py prepare --data ~/masterarbeit/data/4danyone/48Views --frame 10 \
        --out ~/masterarbeit/data/4danyone/diag/48Views_f10_black
    cd ~/masterarbeit/gsplat/examples && python simple_trainer.py default --data_dir <out> \
        --data_factor 1 --result_dir <res> --disable_viewer --disable_video
    python scripts/diag_static_ref.py eval --scene <out> --gsplat-ckpt <res>/ckpts/ckpt_29999_rank0.pt \
        [--own ~/masterarbeit/results/4danyone/48Views_static_f10] [--crops cam18 cam40]
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parent.parent
GSPLAT_EXAMPLES = Path.home() / "masterarbeit" / "gsplat" / "examples"


def prepare(args):
    data, out = Path(args.data).expanduser(), Path(args.out).expanduser()
    if out.exists():
        shutil.rmtree(out)
    (out / "sparse" / "0").mkdir(parents=True)
    for name in ("cameras.txt", "images.txt"):
        shutil.copy(data / "sparse" / "0" / name, out / "sparse" / "0" / name)
    # Startpunkte: Visual Hull dieses Frames (gleiches Koordinatensystem wie sparse/0)
    pts = np.load(data / "points" / f"points3d_frame{args.frame:06d}.npy")
    rgb = np.load(data / "points" / f"colors_frame{args.frame:06d}.npy").round().astype(int)
    with open(out / "sparse" / "0" / "points3D.txt", "w") as f:
        for i, (p, c) in enumerate(zip(pts, rgb), start=1):
            f.write(f"{i} {' '.join(repr(float(x)) for x in p)} {c[0]} {c[1]} {c[2]} 0\n")
    cams = sorted(d.name for d in (data / "images").iterdir() if d.is_dir())
    (out / "images").mkdir()
    (out / "masks").mkdir()
    for cam in cams:
        img = cv2.imread(str(data / "images" / cam / f"{args.frame:05d}.png"), cv2.IMREAD_COLOR)
        m = cv2.imread(str(data / "masks" / cam / f"{args.frame:05d}.png"), cv2.IMREAD_GRAYSCALE)
        comp = (img.astype(np.float32) * (m.astype(np.float32)[..., None] / 255.0)).round().astype(np.uint8)
        cv2.imwrite(str(out / "images" / f"{cam}.png"), comp)
        cv2.imwrite(str(out / "masks" / f"{cam}.png"), m)
    (out / "meta.json").write_text(json.dumps(
        {"source": str(data), "frame": args.frame, "cameras": cams, "points": len(pts),
         "note": "Bild * Maske auf Schwarz, fuer gsplat simple_trainer.py (diag_static_ref.py)"}, indent=1))
    print(f"done: {out} ({len(cams)} Kameras, Frame {args.frame}, {len(pts)} Startpunkte)")


def metrics(pred: torch.Tensor, gt: torch.Tensor, mask, dilate: int) -> dict:
    """pred, gt [H, W, 3] in [0, 1], mask [H, W] uint8 0-255 oder None (Szene ohne Masken)."""
    se = ((pred - gt) ** 2).mean(-1)
    out = {"psnr_full": -10 * np.log10(max(se.mean().item(), 1e-12)), "psnr_person": np.nan, "mae_person": np.nan}
    if mask is not None:
        kernel = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
        region = torch.from_numpy(cv2.dilate((mask > 127).astype(np.uint8), kernel) > 0).cuda()
        ae = (pred - gt).abs().mean(-1) * 255.0
        out["psnr_person"] = -10 * np.log10(max(se[region].mean().item(), 1e-12))
        out["mae_person"] = ae[region].mean().item()
    return out


@torch.no_grad()
def render_gsplat(ckpt_path: Path, scene: Path, test_every: int):
    """Rendert alle Kameras aus einem simple_trainer.py-Checkpoint. -> {cam: (bild, ist_test)}"""
    sys.path.insert(0, str(GSPLAT_EXAMPLES))
    from datasets.colmap import Parser
    from gsplat import rasterization

    parser = Parser(data_dir=str(scene), factor=1, normalize=True, test_every=test_every)
    splats = torch.load(ckpt_path, map_location="cuda", weights_only=False)["splats"]
    colors = torch.cat([splats["sh0"], splats["shN"]], 1)
    sh_degree = int(round(np.sqrt(colors.shape[1]))) - 1
    out = {}
    for i, name in enumerate(parser.image_names):
        cid = parser.camera_ids[i]
        W, H = parser.imsize_dict[cid]
        c2w = torch.from_numpy(parser.camtoworlds[i]).float().cuda()
        K = torch.from_numpy(parser.Ks_dict[cid]).float().cuda()
        img, _, _ = rasterization(
            means=splats["means"], quats=splats["quats"], scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]), colors=colors,
            viewmats=torch.linalg.inv(c2w)[None], Ks=K[None], width=W, height=H,
            sh_degree=sh_degree, near_plane=0.01, far_plane=1e10, packed=False,
        )
        out[Path(name).stem] = (img[0].clamp(0, 1), i % test_every == 0)
    return out, len(splats["means"])


@torch.no_grad()
def render_own(result_dir: Path):
    """Rendert alle Kameras aus einem statischen Lauf des eigenen Trainers (Datensatz mit 1 Frame)."""
    sys.path.insert(0, str(REPO))
    from train import Config, Runner

    raw = yaml.safe_load(open(result_dir / "cfg.yml"))
    fields = Config.__dataclass_fields__
    cfg = Config(**{k: v for k, v in raw.items() if k in fields})
    own_test = set(cfg.test_cams)
    cams = sorted(d.name for d in (Path(cfg.data_dir).expanduser() / "images").iterdir() if d.is_dir())
    cfg.test_cams = cams  # nur Auswertung, alle Kameras laden
    cfg.result_dir = tempfile.mkdtemp(prefix="diag_static_ref_")
    cfg.init_velocity = "zero"
    cfg.lambda_skel = 0.0
    runner = Runner(cfg)
    assert runner.data.num_frames == 1, "erwartet einen statischen Einzelbild-Lauf"
    ckpt = torch.load(result_dir / "ckpts" / "ckpt_final.pt", map_location="cuda", weights_only=False)
    for k in runner.splats.keys():
        runner.splats[k].data = ckpt["splats"][k]
    bg = torch.zeros((1, 3), device="cuda")
    out = {}
    for cam in runner.data.test_cams:
        colors, _, _, _ = runner.render(cam, float(runner.data.times[0]), cfg.sh_degree, bg=bg)
        name = runner.data.cam_names[cam]
        out[name] = (colors[0].clamp(0, 1), name in own_test)
    return out, len(ckpt["splats"]["means"])


def summarize(label: str, rows: list[dict]) -> None:
    for split in ("train", "test"):
        sel = [r for r in rows if r["split"] == split]
        if not sel:
            continue
        m = {k: float(np.nanmean([r[k] for r in sel])) for k in ("psnr_full", "psnr_person", "mae_person")}
        print(f"{label} | {split} ({len(sel)} Kameras) | {m['psnr_full']:.2f} | {m['psnr_person']:.2f} | "
              f"{m['mae_person']:.2f}")


def evaluate(args):
    scene = Path(args.scene).expanduser()
    cams = sorted(p.stem for p in (scene / "images").glob("*.png"))
    gt = {c: torch.from_numpy(cv2.cvtColor(cv2.imread(str(scene / "images" / f"{c}.png")), cv2.COLOR_BGR2RGB))
          .cuda().float() / 255.0 for c in cams}
    # Szenen ohne masks/ (z. B. flame_steak frame0 als Vergleich mit echten Kameras): nur Vollbild
    masks = {c: cv2.imread(str(scene / "masks" / f"{c}.png"), cv2.IMREAD_GRAYSCALE) for c in cams} \
        if (scene / "masks").is_dir() else {c: None for c in cams}

    models = {}
    renders, n = render_gsplat(Path(args.gsplat_ckpt).expanduser(), scene, args.test_every)
    models["gsplat"] = (renders, n)
    if args.own:
        renders, n = render_own(Path(args.own).expanduser())
        models["eigen"] = (renders, n)

    result = {"scene": str(scene), "dilate": args.dilate, "models": {}}
    print("Modell | Kameras | PSNR ganzes Bild | PSNR Person | MAE Person (0–255)")
    for label, (renders, n) in models.items():
        rows = []
        for c in cams:
            img, is_test = renders[c]
            rows.append({"cam": c, "split": "test" if is_test else "train", **metrics(img, gt[c], masks[c], args.dilate)})
        result["models"][label] = {"num_gaussians": n, "per_cam": rows}
        summarize(f"{label} ({n} GS)", rows)
    if len(models) > 1:
        # Kameras, die in beiden Modellen dieselbe Rolle haben
        a, b = (result["models"][k]["per_cam"] for k in ("gsplat", "eigen"))
        for split in ("train", "test"):
            both = [i for i in range(len(cams)) if a[i]["split"] == b[i]["split"] == split]
            if both:
                for label, rows in (("gsplat", a), ("eigen", b)):
                    summarize(f"{label}, gemeinsame Kameras", [rows[i] for i in both])

    out_dir = Path(args.gsplat_ckpt).expanduser().parent.parent
    (out_dir / "static_ref_metrics.json").write_text(json.dumps(result, indent=1))
    # Ausschnitte nebeneinander: Vorlage | gsplat | eigen, um die Person zugeschnitten
    for c in args.crops:
        if masks.get(c) is None:
            continue
        ys, xs = np.nonzero(masks[c] > 127)
        y0, y1 = max(ys.min() - 20, 0), min(ys.max() + 20, masks[c].shape[0])
        x0, x1 = max(xs.min() - 20, 0), min(xs.max() + 20, masks[c].shape[1])
        tiles = [gt[c]] + [renders[c][0] for renders, _ in models.values()]
        row = torch.cat([t[y0:y1, x0:x1] for t in tiles], dim=1)
        img = (row.cpu().numpy() * 255).round().astype(np.uint8)
        cv2.imwrite(str(out_dir / f"crop_{c}_vorlage_{'_'.join(models)}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"gespeichert: {out_dir / 'static_ref_metrics.json'}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--data", required=True, help="flat-Datensatz aus prepare_4danyone.py")
    p.add_argument("--frame", type=int, required=True)
    p.add_argument("--out", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--scene", required=True, help="Ausgabe von prepare")
    e.add_argument("--gsplat-ckpt", required=True)
    e.add_argument("--own", help="Ergebnisordner eines statischen Laufs des eigenen Trainers")
    e.add_argument("--test-every", type=int, default=8, help="wie simple_trainer.py")
    e.add_argument("--dilate", type=int, default=5)
    e.add_argument("--crops", nargs="*", default=["cam18", "cam40"])
    args = ap.parse_args()
    prepare(args) if args.cmd == "prepare" else evaluate(args)


if __name__ == "__main__":
    main()
