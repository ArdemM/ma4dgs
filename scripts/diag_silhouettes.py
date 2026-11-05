"""Silhouetten-Konsistenz der Ansichten einer 4DAnyone-Szene (Kameras aus cameras.json, Masken).

Prüft, ob Posen und Masken der Kameras zueinander passen, ohne Training. Für jeden Frame und
jede Kamera c:
  - H_-c: Visual Hull aus allen anderen Kameras (Space Carving wie prepare_4danyone.py,
    Toleranz 0), auf einem um --pad erweiterten Gitter mit Voxel = längste Seite / --res,
  - Deckung: Anteil der Maskenpixel von c, deren Sehstrahl H_-c trifft (Abtastung mit halber
    Voxelgröße). Bei passenden Posen und Masken nahe 1; kleiner heißt, c sieht Person, wo die
    übrigen Kameras keine zulassen,
  - Saumbreite: fehlender Maskenanteil / Umfang der Maske in Pixeln, also die Breite eines
    gleichmäßigen Randstreifens mit derselben Fläche,
  - Veto: Anteil von H_-c, den c allein wegschneidet (c sagt Hintergrund, alle anderen Person),
  - IoU zwischen Maske und Projektion von H_-c, und der ganzzahlige Bildversatz (±--max-shift px),
    bei dem die Maske am besten zur Projektion passt. Bei passenden Posen 0.
Diskretisierung: Auch bei exakt passenden Masken fehlt am Rand ein Streifen in der Größenordnung
eines halben Voxels (0,75 cm sind bei 4DAnyone ~4,5 px). Deshalb mit mehreren --res rechnen: Was
mit kleineren Voxeln verschwindet, ist Diskretisierung; was bleibt, ist Inkonsistenz.
--synthetic prüft zusätzlich die Umsetzung mit Masken, die aus dem Hull selbst gerendert sind
(Erwartung: Deckung 1); die Diskretisierung erfasst das nicht.

Ausgabe: Tabelle pro Frame, JSON und Fehlerbilder der auffälligsten Kameras
(rot: Maske ohne Hull der anderen, blau: Hull der anderen ohne Maske) in --out.

Aufruf: python scripts/diag_silhouettes.py ~/masterarbeit/data/4danyone/48Views --frames 10 60 --res 128 256 512
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_4danyone import Hull, load_cameras  # noqa: E402


def project_fg(hull: Hull, masks: torch.Tensor, cams: list[int], pts: torch.Tensor) -> torch.Tensor:
    """pts [M, 3] -> [len(cams), M] bool: Punkt liegt im Bild der Kamera und auf deren Maske.

    Pixel i deckt [i, i + 1) ab (Mitte bei i + 0,5, wie COLMAP/gsplat; cameras.json hat cx = W/2),
    daher floor. prepare_4danyone.py rundet, das verschiebt die Abtastung um bis zu 0,5 px.
    """
    w2c, K = hull.w2c[cams], hull.K[cams]
    cam = pts @ w2c[:, :3, :3].transpose(1, 2) + w2c[:, None, :3, 3]
    z = cam[..., 2]
    uv = (cam @ K.transpose(1, 2))[..., :2] / z.clamp(min=1e-6)[..., None]
    u, v = uv[..., 0].floor().long(), uv[..., 1].floor().long()
    inside = (z > 0) & (u >= 0) & (u < hull.W) & (v >= 0) & (v < hull.H)
    idx = v.clamp(0, hull.H - 1) * hull.W + u.clamp(0, hull.W - 1)
    return inside & (torch.gather(masks[cams].reshape(len(cams), -1), 1, idx) >= hull.thresh)


def grid_count(hull: Hull, masks: torch.Tensor, cams: list[int], lo: np.ndarray, voxel: float, dims: tuple):
    """[X, Y, Z] int16: Anzahl der Kameras aus cams, deren Maske die Voxelmitte enthält."""
    axes = [torch.arange(n, device="cuda", dtype=torch.float32) * voxel + float(l) + voxel / 2
            for n, l in zip(dims, lo)]
    gy, gz = torch.meshgrid(axes[1], axes[2], indexing="ij")
    out = torch.empty(dims, dtype=torch.int16, device="cuda")
    for i in range(dims[0]):  # Scheibe für Scheibe, um Speicher zu sparen
        pts = torch.stack([torch.full_like(gy, axes[0][i]), gy, gz], dim=-1).reshape(-1, 3)
        out[i] = project_fg(hull, masks, cams, pts).sum(0, dtype=torch.int16).reshape(dims[1], dims[2])
    return out


@torch.no_grad()
def render_occ(occ: torch.Tensor, lo: np.ndarray, voxel: float, cam: dict, rect=None, chunk: int = 8192):
    """Belegung [X, Y, Z] -> Bild [H, W] bool: Sehstrahl des Pixels trifft ein belegtes Voxel.

    rect = (u0, u1, v0, v1) begrenzt die Strahlen auf einen Bildausschnitt (Standard: Projektion des Gitters).
    """
    dev = "cuda"
    c2w = torch.tensor(cam["c2w"], dtype=torch.float32, device=dev)
    K = torch.tensor(cam["K"], dtype=torch.float32, device=dev)
    W, H = cam["W"], cam["H"]
    dims = torch.tensor(occ.shape, device=dev)
    lo_t = torch.tensor(lo, dtype=torch.float32, device=dev)
    hi_t = lo_t + dims.float() * voxel
    corners = torch.stack(torch.meshgrid(*[torch.stack([lo_t[k], hi_t[k]]) for k in range(3)], indexing="ij"),
                          dim=-1).reshape(-1, 3)
    w2c = torch.linalg.inv(c2w)
    cc = corners @ w2c[:3, :3].T + w2c[:3, 3]
    z = cc[:, 2]
    assert (z > 0).all(), "Kamera liegt im Gitter"
    if rect is None:
        uv = (cc @ K.T)[:, :2] / z[:, None]
        rect = (int(uv[:, 0].min().floor().clamp(0, W - 1)), int(uv[:, 0].max().ceil().clamp(0, W - 1)),
                int(uv[:, 1].min().floor().clamp(0, H - 1)), int(uv[:, 1].max().ceil().clamp(0, H - 1)))
    u0, u1, v0, v1 = rect
    ts = torch.arange(float(z.min()), float(z.max()), voxel / 2, device=dev)  # Tiefen, Schritt halbe Voxelgröße
    vs, us = torch.meshgrid(torch.arange(v0, v1 + 1, device=dev), torch.arange(u0, u1 + 1, device=dev), indexing="ij")
    pix = torch.stack([us.flatten(), vs.flatten()], dim=-1)
    dirs = torch.cat([pix.float() + 0.5, torch.ones_like(pix[:, :1], dtype=torch.float32)], dim=-1) @ torch.linalg.inv(K).T
    dirs = dirs @ c2w[:3, :3].T  # Tiefe 1 entlang der optischen Achse
    flat = occ.flatten()
    img = torch.zeros(H * W, dtype=torch.bool, device=dev)
    for i in range(0, len(pix), chunk):
        p = c2w[:3, 3] + dirs[i:i + chunk, None, :] * ts[None, :, None]
        ijk = ((p - lo_t) / voxel).floor().long()
        valid = ((ijk >= 0) & (ijk < dims)).all(-1)
        idx = (ijk[..., 0] * dims[1] + ijk[..., 1]) * dims[2] + ijk[..., 2]
        hit = (flat[idx.clamp(0, len(flat) - 1)] & valid).any(1)
        img[pix[i:i + chunk, 1] * W + pix[i:i + chunk, 0]] = hit
    return img.reshape(H, W)


def mask_rect(m: torch.Tensor, margin: int):
    ys, xs = torch.nonzero(m, as_tuple=True)
    H, W = m.shape
    return (max(int(xs.min()) - margin, 0), min(int(xs.max()) + margin, W - 1),
            max(int(ys.min()) - margin, 0), min(int(ys.max()) + margin, H - 1))


def best_shift(m: torch.Tensor, p: torch.Tensor, max_shift: int):
    """Ganzzahliger Versatz (dx, dy) der Maske m, der die IoU mit p maximiert."""
    ys, xs = torch.nonzero(m | p, as_tuple=True)
    y0, y1 = max(int(ys.min()) - max_shift, 0), int(ys.max()) + max_shift + 1
    x0, x1 = max(int(xs.min()) - max_shift, 0), int(xs.max()) + max_shift + 1
    mc, pc = m[y0:y1, x0:x1], p[y0:y1, x0:x1]
    best = (-1.0, 0, 0)
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            ms = torch.roll(mc, shifts=(dy, dx), dims=(0, 1))
            iou = float((ms & pc).sum()) / max(float((ms | pc).sum()), 1.0)
            best = max(best, (iou, dx, dy))
    return best


def perimeter(m: np.ndarray) -> float:
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return float(sum(cv2.arcLength(c, True) for c in cs))


def consistency(hull: Hull, cams: list[dict], masks: torch.Tensor, lo, voxel, dims, args, keep_proj=()):
    """Kennzahlen pro Kamera für einen Maskensatz; Projektionen von H_-c für die Kameras in keep_proj."""
    C = len(cams)
    votes = grid_count(hull, masks, list(range(C)), lo, voxel, dims)
    rows, projs = [], {}
    for c in range(C):
        fc = grid_count(hull, masks, [c], lo, voxel, dims)
        others = (votes - fc) == C - 1
        M = masks[c] >= hull.thresh
        P = render_occ(others, lo, voxel, cams[c], rect=mask_rect(M, args.margin))
        inter, area = float((M & P).sum()), float(M.sum())
        iou_s, dx, dy = best_shift(M, P, args.max_shift)
        rows.append({
            "cam": cams[c]["name"],
            "coverage": inter / max(area, 1.0),
            "band_px": (area - inter) / max(perimeter(M.cpu().numpy()), 1.0),
            "veto": float((others & (fc == 0)).sum()) / max(int(others.sum()), 1),
            "iou": inter / max(float((M | P).sum()), 1.0),
            "iou_shifted": iou_s, "shift_dx": dx, "shift_dy": dy,
            "mask_on_border": bool(M[0].any() or M[-1].any() or M[:, 0].any() or M[:, -1].any()),
        })
        if cams[c]["name"] in keep_proj:
            projs[cams[c]["name"]] = P.cpu().numpy()
        del fc, others
    return rows, votes == C, projs


def print_table(rows: list[dict], ref: list[dict] | None):
    print("Kamera | Deckung | Saum px | Veto | IoU | bester Versatz (dx, dy) -> IoU | Maske am Bildrand"
          + (" | Deckung synthetisch" if ref else ""))
    for i in np.argsort([r["coverage"] for r in rows]):
        r = rows[i]
        print(f"{r['cam']} | {r['coverage']:.3f} | {r['band_px']:.2f} | {r['veto']:.3f} | {r['iou']:.3f} | "
              f"({r['shift_dx']:+d}, {r['shift_dy']:+d}) -> {r['iou_shifted']:.3f} | "
              f"{'ja' if r['mask_on_border'] else ''}" + (f" | {ref[i]['coverage']:.3f}" if ref else ""))


def summary(label: str, rows: list[dict]) -> str:
    cov = np.array([r["coverage"] for r in rows])
    band = np.array([r["band_px"] for r in rows])
    veto = np.array([r["veto"] for r in rows])
    shift = np.array([max(abs(r["shift_dx"]), abs(r["shift_dy"])) for r in rows])
    return (f"{label}: Deckung Median {np.median(cov):.3f} (min {cov.min():.3f}), Saum Median "
            f"{np.median(band):.2f} px (max {band.max():.2f}), Veto Median {np.median(veto):.3f} "
            f"(max {veto.max():.3f}), Versatz >= 2 px: {int((shift >= 2).sum())} von {len(rows)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="flat-Datensatz aus prepare_4danyone.py (meta.json mit source)")
    ap.add_argument("--frames", type=int, nargs="+", default=[10, 60])
    ap.add_argument("--res", type=int, nargs="+", default=[256], help="Voxel = längste Seite / res; mehrere Werte möglich")
    ap.add_argument("--src", help="4DAnyone-Ausgabe mit cameras.json (Standard: source aus meta.json)")
    ap.add_argument("--pad", type=float, default=0.15, help="Erweiterung des Gitters je Seite (Anteil)")
    ap.add_argument("--margin", type=int, default=40, help="Strahlen nur im Maskenrechteck ± margin px")
    ap.add_argument("--max-shift", type=int, default=8)
    ap.add_argument("--synthetic", action="store_true", help="zusätzlich mit aus dem Hull gerenderten Masken")
    ap.add_argument("--worst", type=int, default=3, help="Fehlerbilder für so viele Kameras speichern")
    ap.add_argument("--quiet", action="store_true", help="keine Tabelle pro Kamera")
    ap.add_argument("--out", help="Standard: ~/masterarbeit/results/4danyone/diag_silhouettes/<name>")
    args = ap.parse_args()

    data = Path(args.data).expanduser()
    meta = json.loads((data / "meta.json").read_text())
    cams = load_cameras(Path(args.src or meta["source"]))
    assert [c["name"] for c in cams] == meta["cameras"], "Kameras passen nicht zum Datensatz"
    thresh = meta.get("hull", {}).get("mask_thresh", 0.5)
    hull = Hull(cams, thresh, tolerance=0)
    out = Path(args.out).expanduser() if args.out else \
        Path.home() / "masterarbeit" / "results" / "4danyone" / "diag_silhouettes" / data.name
    out.mkdir(parents=True, exist_ok=True)

    result = {"data": str(data), "pad": args.pad, "runs": []}
    for f in args.frames:
        masks = torch.from_numpy(np.stack([
            cv2.imread(str(data / "masks" / c["name"] / f"{f:05d}.png"), cv2.IMREAD_GRAYSCALE) for c in cams
        ])).cuda().float() / 255.0
        # Gitter wie prepare_4danyone.py (Grobdurchgang, Voxel = längste Seite / res), dann erweitert
        occ, origin, v96 = hull.carve(masks, *hull.box, 96)
        nz = occ.nonzero()
        lo0 = origin + (nz.min(0).values.cpu().numpy() - 2) * v96
        hi0 = origin + (nz.max(0).values.cpu().numpy() + 3) * v96
        for res in args.res:
            voxel = float((hi0 - lo0).max()) / res
            lo, hi = lo0 - args.pad * (hi0 - lo0), hi0 + args.pad * (hi0 - lo0)
            dims = tuple(int(n) for n in np.ceil((hi - lo) / voxel))
            print(f"\n== Frame {f}, res {res}: Voxel {voxel * 100:.2f} cm, Gitter {dims}", flush=True)
            real, hull_all, _ = consistency(hull, cams, masks, lo, voxel, dims, args)
            worst = [real[i]["cam"] for i in np.argsort([r["coverage"] for r in real])[:args.worst]]
            synth = None
            if args.synthetic:
                synth_masks = torch.stack([render_occ(hull_all, lo, voxel, c) for c in cams]).float()
                synth, _, _ = consistency(hull, cams, synth_masks, lo, voxel, dims, args)
                del synth_masks
            if not args.quiet:
                print_table(real, synth)
            print("Summe " + summary("echt", real))
            if synth:
                print("Summe " + summary("synthetisch", synth))
            result["runs"].append({"frame": f, "res": res, "voxel_m": voxel, "grid": dims, "real": real,
                                   "synthetic": synth})

            # Fehlerbilder: rot = Maske ohne H_-c, blau = H_-c ohne Maske, auf dem Kamerabild
            if args.worst:
                C = len(cams)
                votes = grid_count(hull, masks, list(range(C)), lo, voxel, dims)
                for name in worst:
                    c = next(i for i, cam in enumerate(cams) if cam["name"] == name)
                    others = (votes - grid_count(hull, masks, [c], lo, voxel, dims)) == C - 1
                    M_t = masks[c] >= thresh
                    P = render_occ(others, lo, voxel, cams[c], rect=mask_rect(M_t, args.margin)).cpu().numpy()
                    M = M_t.cpu().numpy()
                    img = cv2.imread(str(data / "images" / name / f"{f:05d}.png")).astype(np.float32) * 0.5
                    img[M & ~P] = (0, 0, 255)
                    img[P & ~M] = (255, 80, 0)
                    cv2.imwrite(str(out / f"f{f:03d}_res{res}_{name}_fehler.png"), img.astype(np.uint8))
                del votes
            del hull_all
            torch.cuda.empty_cache()
        del masks

    (out / "silhouettes.json").write_text(json.dumps(result, indent=1))
    print(f"\ngespeichert: {out}")


if __name__ == "__main__":
    main()
