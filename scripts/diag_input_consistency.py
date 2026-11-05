"""Zeitliche Konsistenz der Eingabevideos (Diagnose D2, Ansaetze_schnelle_Bewegung.md).

Frage: Ändern die generierten Ansichten beim Springen ihr Aussehen von Frame zu Frame
stärker, als die Bewegung erklärt? Solche Änderungen könnte ein 4D-Modell nicht mit
bewegten Gaußschen über mehrere Frames teilen.

Pro Kamera und Frame-Paar f → f+1 (halbe Auflösung):
  - optischer Fluss vorwärts und rückwärts (RAFT, torchvision),
  - Warp-Residuum |I_f(x) − I_{f+1}(x + F(x))| (Mittel über RGB, 0–255) auf Pixeln in der
    erodierten Maske, deren Ziel ebenfalls in der Maske liegt und die die Vorwärts-
    Rückwärts-Prüfung bestehen (|F + B(x+F)|² < 0,01·(|F|² + |B|²) + 0,5, Sundaram et al.
    2010). Verdeckte oder unzuverlässig verfolgte Pixel fallen so heraus; ihr Anteil
    wird als "nicht verfolgbar" mitgeschrieben,
  - Median des Flussbetrags in der Maske (Pixel voller Auflösung),
  - Residuum je Flussbetrag-Klasse (Pixel), damit gleiche Bewegung verglichen wird.
Pro Kamera und Frame: Schärfe (Varianz des Laplace-Bilds in der erodierten Maske, volle
Auflösung) und Farbmittel in der Maske.

Referenz: das echte Quellvideo (conditioning/source.mkv, Pexels-Original) wie bei 4DAnyone
zugeschnitten (links 24 px weg, Breite 2112) und um den Faktor 3 verkleinert, Masken mit
demselben BiRefNet wie scripts/prepare_4danyone.py. Weil die generierten Videos h264 mit
~3 Mbit/s sind, das Quellvideo aber verlustfrei, wird es zusätzlich mit libx264 auf dieselbe
Bitrate kodiert ("source_h264"). Das Quellvideo zeigt eine andere Ansicht und echte
Bewegungsunschärfe; es ist ein Anhaltspunkt für reale Videos, keine Ground Truth.

Ausgabe: <out>/consistency.json (alle Werte pro Kamera) und eine Übersicht auf stdout.
Aufruf: python scripts/diag_input_consistency.py --data-dir <szene> \
            --source <4DAnyone-Ausgabe>/conditioning/source.mkv --out <ordner>
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as Fn
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_4danyone import BIREFNET_REPO, BIREFNET_REVISION  # noqa: E402

# Klassen des Flussbetrags in Pixeln voller Auflösung
FLOW_BINS = [0, 1, 2, 4, 8, 16, 32, 1e9]
PHASES = {"ruhig": (0, 39), "Sprung": (50, 70)}  # Frame-Bereiche; Paar f→f+1 zählt, wenn beide darin liegen
HIGH_RESIDUAL = 10.0  # Graustufen (0–255): Schwelle für "stark abweichende" Pixel


def load_dataset(data: Path, cam: str, num_frames: int):
    def read(f):
        img = cv2.cvtColor(cv2.imread(str(data / "images" / cam / f"{f:05d}.png")), cv2.COLOR_BGR2RGB)
        m = cv2.imread(str(data / "masks" / cam / f"{f:05d}.png"), cv2.IMREAD_GRAYSCALE)
        return img, m

    with ThreadPoolExecutor(8) as pool:
        out = list(pool.map(read, range(num_frames)))
    return np.stack([o[0] for o in out]), np.stack([o[1] for o in out])


def read_video(path: Path, crop_x: int, crop_w: int, size: tuple) -> np.ndarray:
    """Video -> [F, H, W, 3] uint8 RGB, zugeschnitten (x, Breite) und auf size = (W, H) verkleinert."""
    W, H = size
    vf = f"crop={crop_w}:in_h:{crop_x}:0,scale={W}:{H}:flags=area"
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3).copy()


def h264_roundtrip(frames: np.ndarray, bitrate: str, fps: int = 25) -> np.ndarray:
    """Bilder mit libx264 (High, yuv420p, Zielbitrate) kodieren und wieder dekodieren."""
    F, H, W, _ = frames.shape
    enc = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
         "-i", "-", "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p", "-b:v", bitrate,
         "-f", "matroska", "-"],
        input=frames.tobytes(), check=True, capture_output=True,
    ).stdout
    dec = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "matroska", "-i", "-", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        input=enc, check=True, capture_output=True,
    ).stdout
    return np.frombuffer(dec, np.uint8).reshape(F, H, W, 3).copy()


@torch.no_grad()
def birefnet_masks(frames: np.ndarray, batch: int = 4) -> np.ndarray:
    """Vordergrundmasken 0–255 wie scripts/prepare_4danyone.py (BiRefNet, 1024², Revision fest)."""
    from transformers import AutoModelForImageSegmentation

    model = AutoModelForImageSegmentation.from_pretrained(
        BIREFNET_REPO, trust_remote_code=True, revision=BIREFNET_REVISION
    ).cuda().eval().half()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    H, W = frames.shape[1:3]
    out = []
    for i in range(0, len(frames), batch):
        x = torch.from_numpy(frames[i:i + batch]).cuda().permute(0, 3, 1, 2).float() / 255.0
        x = Fn.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False)
        pred = model(((x - mean) / std).half())[-1].sigmoid().float()
        pred = Fn.interpolate(pred, size=(H, W), mode="bilinear", align_corners=False)
        out.append((pred[:, 0].clamp(0, 1) * 255).round().byte().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out)


def warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """img [B, C, H, W] an x + flow(x) abtasten (bilinear, außerhalb 0)."""
    B, _, H, W = img.shape
    ys, xs = torch.meshgrid(
        torch.arange(H, device=img.device, dtype=torch.float32),
        torch.arange(W, device=img.device, dtype=torch.float32),
        indexing="ij",
    )
    gx = 2.0 * (xs + flow[:, 0]) / (W - 1) - 1.0
    gy = 2.0 * (ys + flow[:, 1]) / (H - 1) - 1.0
    return Fn.grid_sample(img, torch.stack([gx, gy], -1), mode="bilinear", padding_mode="zeros", align_corners=True)


class Analyzer:
    def __init__(self, scale: float, iters: int, erode_px: int, batch: int):
        weights = Raft_Large_Weights.DEFAULT
        self.model = raft_large(weights=weights, progress=False).cuda().eval()
        self.tf = weights.transforms()
        self.scale, self.iters, self.batch = scale, iters, batch
        self.kernel = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)

    @torch.no_grad()
    def sequence(self, frames: np.ndarray, masks: np.ndarray) -> dict:
        """frames [F, H, W, 3] uint8, masks [F, H, W] uint8 -> Werte pro Paar und pro Frame."""
        F, H, W, _ = frames.shape
        Hs, Ws = int(H * self.scale) // 8 * 8, int(W * self.scale) // 8 * 8
        up = W / Ws  # Pixel halbe -> volle Auflösung (x und y gleich bei 704×1280 und 0,5)

        # Pro Frame: Schärfe und Farbe in der erodierten Maske (volle Auflösung)
        eroded = np.stack([cv2.erode((m > 127).astype(np.uint8), self.kernel) > 0 for m in masks])
        sharp, color = [], []
        for img, m in zip(frames, eroded):
            lap = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32), cv2.CV_32F)
            sharp.append(float(lap[m].var()) if m.any() else float("nan"))
            color.append(img[m].mean(0).tolist() if m.any() else [float("nan")] * 3)

        img_s = torch.from_numpy(frames).permute(0, 3, 1, 2).float().cuda() / 255.0
        img_s = Fn.interpolate(img_s, (Hs, Ws), mode="area")
        msk_s = Fn.interpolate(torch.from_numpy(masks).float().cuda()[:, None] / 255.0, (Hs, Ws), mode="area")
        er_s = Fn.interpolate(torch.from_numpy(eroded).float().cuda()[:, None], (Hs, Ws), mode="area") > 0.99

        res, res_all, high, flow_med, tracked = [], [], [], [], []
        bin_sum = np.zeros((F - 1, len(FLOW_BINS) - 1))
        bin_cnt = np.zeros((F - 1, len(FLOW_BINS) - 1))
        edges = torch.tensor(FLOW_BINS, device="cuda", dtype=torch.float32)
        for i in range(0, F - 1, self.batch):
            j = min(i + self.batch, F - 1)
            a, b = img_s[i:j], img_s[i + 1:j + 1]
            x1, x2 = self.tf(torch.cat([a, b]), torch.cat([b, a]))
            flows = self.model(x1, x2, num_flow_updates=self.iters)[-1]
            fw, bw = flows[: j - i], flows[j - i:]
            bw_at = warp(bw, fw)
            cons = (fw + bw_at).pow(2).sum(1) < 0.01 * (fw.pow(2).sum(1) + bw_at.pow(2).sum(1)) + 0.5
            m0 = er_s[i:j, 0]
            m1 = warp(msk_s[i + 1:j + 1], fw)[:, 0] > 0.5
            valid = m0 & m1 & cons
            r = (a - warp(b, fw)).abs().mean(1) * 255.0  # [B, Hs, Ws]
            mag = fw.norm(dim=1) * up
            for k in range(j - i):
                v, m = valid[k], m0[k]
                both = m & m1[k]  # ohne Vorwärts-Rückwärts-Prüfung
                n_m = int(m.sum())
                res.append(float(r[k][v].mean()) if v.any() else float("nan"))
                res_all.append(float(r[k][both].mean()) if both.any() else float("nan"))
                high.append(float((r[k][both] > HIGH_RESIDUAL).float().mean()) if both.any() else float("nan"))
                flow_med.append(float(mag[k][m].median()) if n_m else float("nan"))
                tracked.append(float(v.sum()) / max(n_m, 1))
                idx = torch.bucketize(mag[k][v], edges[1:-1])
                bin_sum[i + k] = torch.zeros(len(FLOW_BINS) - 1, device="cuda").index_add_(0, idx, r[k][v]).cpu().numpy()
                bin_cnt[i + k] = torch.bincount(idx, minlength=len(FLOW_BINS) - 1).cpu().numpy()
        return {
            "residual": res, "residual_all": res_all, "high_frac": high,
            "flow_median_px": flow_med, "tracked_frac": tracked,
            "bin_sum": bin_sum.tolist(), "bin_count": bin_cnt.tolist(),
            "sharpness": sharp, "color": color,
        }


def phase_pairs(name, F):
    f0, f1 = PHASES[name]
    return [f for f in range(F - 1) if f0 <= f and f + 1 <= f1]


def summarize(results: dict, F: int) -> None:
    gen = {k: v for k, v in results.items() if k.startswith("cam")}
    refs = {k: v for k, v in results.items() if not k.startswith("cam")}

    def stat(seqs, key, pairs, agg=np.nanmean):
        # Mittel über Paare je Sequenz, dann Median über Sequenzen
        return float(np.nanmedian([agg([s[key][f] for f in pairs]) for s in seqs.values()]))

    print(f"\n== Pro Phase (generiert: Median über {len(gen)} Kameras des Mittels über die Paare)")
    print("Residuum = verfolgbare Pixel; ohne Prüfung = alle Pixel mit Ziel in der Maske; > 10 = Anteil davon")
    print("Quelle        | Phase  | Residuum | ohne Prüfung | > 10 | Fluss (Median px) | verfolgbar | "
          "Schärfe (Laplace-Var.) | Farbsprung |ΔRGB|")
    for label, seqs in [("generiert", gen)] + [(k, {k: v}) for k, v in refs.items()]:
        for ph in PHASES:
            pairs = phase_pairs(ph, F)
            frames = range(PHASES[ph][0], PHASES[ph][1] + 1)
            sharp = float(np.nanmedian([np.nanmean([s["sharpness"][f] for f in frames]) for s in seqs.values()]))
            jump = float(np.nanmedian([
                np.nanmean([np.abs(np.subtract(s["color"][f + 1], s["color"][f])).mean() for f in pairs])
                for s in seqs.values()
            ]))
            print(f"{label:13s} | {ph:6s} | {stat(seqs, 'residual', pairs):8.2f} | "
                  f"{stat(seqs, 'residual_all', pairs):12.2f} | {stat(seqs, 'high_frac', pairs) * 100:3.1f}% | "
                  f"{stat(seqs, 'flow_median_px', pairs):17.2f} | {stat(seqs, 'tracked_frac', pairs):10.3f} | "
                  f"{sharp:22.1f} | {jump:10.2f}")

    print("\n== Residuum je Flussbetrag (Pixel voller Auflösung), Pixel aller Paare der Phase gepoolt")
    hdr = " | ".join(f"{FLOW_BINS[b]:g}–{FLOW_BINS[b + 1]:g}" if FLOW_BINS[b + 1] < 1e8 else f">{FLOW_BINS[b]:g}"
                     for b in range(len(FLOW_BINS) - 1))
    print(f"Quelle        | Phase  | {hdr}")
    for label, seqs in [("generiert", gen)] + [(k, {k: v}) for k, v in refs.items()]:
        for ph in PHASES:
            pairs = phase_pairs(ph, F)
            s_ = sum(np.asarray(s["bin_sum"])[pairs].sum(0) for s in seqs.values())
            c_ = sum(np.asarray(s["bin_count"])[pairs].sum(0) for s in seqs.values())
            cells = [f"{s_[b] / c_[b]:5.2f} ({c_[b] / c_.sum() * 100:4.1f}%)" if c_[b] > 100 else "    –" for b in range(len(c_))]
            print(f"{label:13s} | {ph:6s} | " + " | ".join(cells))

    print("\n== Latent-Grenzen (Wan2.2: Frame 0 allein, dann Gruppen 1–4, 5–8, …; Paar f→f+1 überquert eine")
    print("   Grenze, wenn f mod 4 = 0). Residuum geteilt durch das Residuum desselben Paars im Mittel der")
    print("   Nachbarpaare f−1 und f+1, gemittelt je f mod 4")
    for label, seqs in [("generiert", gen)] + [(k, {k: v}) for k, v in refs.items()]:
        for ph in PHASES:
            pairs = [f for f in phase_pairs(ph, F) if 1 <= f < F - 2]
            out = []
            for k in range(4):
                ratios = []
                for s in seqs.values():
                    r = s["residual"]
                    ratios += [r[f] / (0.5 * (r[f - 1] + r[f + 1])) for f in pairs if f % 4 == k]
                out.append(f"f mod 4 = {k}: {np.nanmean(ratios):.3f}")
            print(f"{label:13s} | {ph:6s} | " + " | ".join(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--source", type=Path, default=None, help="conditioning/source.mkv (optional)")
    ap.add_argument("--source-crop", type=int, nargs=2, default=[24, 2112], help="x, Breite des 4DAnyone-Zuschnitts")
    ap.add_argument("--source-bitrate", default="3052k", help="Bitrate der generierten Videos (ffprobe)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cams", nargs="*", default=None, help="Standard: alle")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--erode", type=int, default=5, help="Erosion der Maske in Pixeln voller Auflösung")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    data = args.data_dir.expanduser()
    args.out.mkdir(parents=True, exist_ok=True)
    cams = args.cams or sorted(p.name for p in (data / "images").iterdir())
    F = len(list((data / "images" / cams[0]).glob("*.png")))
    an = Analyzer(args.scale, args.iters, args.erode, args.batch)
    results = {}

    if args.source is not None:
        W, H = cv2.imread(str(data / "images" / cams[0] / "00000.png")).shape[1::-1]
        src = read_video(args.source, *args.source_crop, (W, H))[:F]
        assert len(src) == F, f"Quellvideo hat {len(src)} Frames, Szene {F}"
        masks = birefnet_masks(src)
        np.save(args.out / "source_masks.npy", masks)
        results["source"] = an.sequence(src, masks)
        results["source_h264"] = an.sequence(h264_roundtrip(src, args.source_bitrate), masks)
        print("Quellvideo fertig", flush=True)

    for i, cam in enumerate(cams):
        frames, masks = load_dataset(data, cam, F)
        results[cam] = an.sequence(frames, masks)
        if i % 8 == 0:
            print(f"{cam} fertig ({i + 1}/{len(cams)})", flush=True)

    meta = {"data_dir": str(data), "source": str(args.source), "num_frames": F, "phases": PHASES,
            "flow_bins_px": FLOW_BINS, "scale": args.scale, "erode_px": args.erode, "iters": args.iters,
            "source_crop": args.source_crop, "source_bitrate": args.source_bitrate}
    (args.out / "consistency.json").write_text(json.dumps({"meta": meta, "results": results}))
    summarize(results, F)


if __name__ == "__main__":
    main()
