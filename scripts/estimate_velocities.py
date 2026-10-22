"""Startgeschwindigkeit pro Hull-Punkt aus optischem Fluss (RAFT, torchvision).

Hintergrund: Die kNN-Zuordnung zwischen den Hull-Punktwolken zweier Frames
(FreeTimeGS Abschnitt 3.2) findet bei Bewegung entlang der eigenen Oberfläche
(Körper springt nach oben, Rumpf und Beine sind senkrechte Flächen) den Punkt
auf derselben Fläche und unterschätzt die Bewegung um eine Größenordnung
(Seilspringen Frame 60: Kopf +3,9 cm, kNN-Fluss +0,3 cm). Die Bilder enthalten
dagegen Textur. Deshalb hier: 2D-Fluss pro Kamera zwischen Frame f und f+1,
dann je Punkt die 3D-Verschiebung d per Kleinste-Quadrate über alle Kameras,
die den Punkt sehen (Linearisierung der Projektion, robust per IRLS/Huber).

Eingabe: vorbereitete Szene (scripts/prepare_4danyone.py): images/, masks/,
points/points3d_frameNNNNNN.npy, sparse/0 (PINHOLE, Rohkoordinaten).
Ausgabe: points/velocities_frameNNNNNN.npy [M, 3], Rohkoordinaten pro
normierter Zeiteinheit (wie knn_velocities), dazu points/velocities_stats.json.
Training: train.py --init-velocity file (ma4dgs/data.py dreht und skaliert mit).

Aufruf: python scripts/estimate_velocities.py --data-dir <scene> [--exclude-cams cam19 cam40]
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as Fn
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large


def qvec_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def load_colmap(sparse: Path):
    cams = {}
    for line in (sparse / "cameras.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        t = line.split()
        assert t[1] == "PINHOLE", t
        W, H, fx, fy, cx, cy = int(t[2]), int(t[3]), *map(float, t[4:8])
        cams[int(t[0])] = dict(W=W, H=H, K=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]))
    views = []
    for line in (sparse / "images.txt").read_text().splitlines():
        t = line.split()
        if len(t) != 10 or line.startswith("#"):  # Bildzeilen haben 10 Felder, Punktzeilen sind leer
            continue
        q, tr = np.array(list(map(float, t[1:5]))), np.array(list(map(float, t[5:8])))
        R = qvec_to_rotmat(q)
        w2c = np.eye(4); w2c[:3, :3] = R; w2c[:3, 3] = tr
        views.append(dict(name=Path(t[9]).stem, w2c=w2c, **cams[int(t[8])]))
    views.sort(key=lambda v: v["name"])
    return views


class FlowSolver:
    def __init__(self, views, device, scale, iters):
        self.views = views
        self.dev = device
        self.scale = scale
        self.iters = iters
        self.W, self.H = views[0]["W"], views[0]["H"]
        self.w2c = torch.tensor(np.stack([v["w2c"] for v in views]), dtype=torch.float32, device=device)
        self.K = torch.tensor(np.stack([v["K"] for v in views]), dtype=torch.float32, device=device)
        weights = Raft_Large_Weights.DEFAULT
        self.model = raft_large(weights=weights, progress=False).to(device).eval()
        self.tf = weights.transforms()

    def read(self, data, kind, cam, frame):
        p = data / kind / cam / f"{frame:05d}.png"
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE if kind == "masks" else cv2.IMREAD_COLOR)
        assert img is not None, p
        if kind == "images":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    @torch.no_grad()
    def flows(self, data, f0, f1, batch=4):
        """Fluss von Frame f0 nach f1 für alle Kameras: [C, 2, H, W] in Pixeln voller Auflösung."""
        out = []
        Hs, Ws = int(self.H * self.scale) // 8 * 8, int(self.W * self.scale) // 8 * 8
        names = [v["name"] for v in self.views]
        for i in range(0, len(names), batch):
            a = torch.stack([torch.from_numpy(self.read(data, "images", n, f0)) for n in names[i:i + batch]])
            b = torch.stack([torch.from_numpy(self.read(data, "images", n, f1)) for n in names[i:i + batch]])
            a = a.permute(0, 3, 1, 2).float().to(self.dev) / 255.0
            b = b.permute(0, 3, 1, 2).float().to(self.dev) / 255.0
            a = Fn.interpolate(a, (Hs, Ws), mode="bilinear", align_corners=False)
            b = Fn.interpolate(b, (Hs, Ws), mode="bilinear", align_corners=False)
            a, b = self.tf(a, b)
            flow = self.model(a, b, num_flow_updates=self.iters)[-1]  # [B, 2, Hs, Ws]
            flow = Fn.interpolate(flow, (self.H, self.W), mode="bilinear", align_corners=False)
            flow[:, 0] *= self.W / Ws
            flow[:, 1] *= self.H / Hs
            out.append(flow)
        return torch.cat(out)

    def project(self, pts):
        cam = pts @ self.w2c[:, :3, :3].transpose(1, 2) + self.w2c[:, None, :3, 3]  # [C, M, 3]
        z = cam[..., 2]
        uv = (cam @ self.K.transpose(1, 2))[..., :2] / z.clamp(min=1e-6)[..., None]
        return cam, uv, z

    @torch.no_grad()
    def solve(self, pts, flow, mask0, mask1, voxel, min_cams=3, huber_px=3.0, cell=4):
        """pts [M, 3] (Rohkoordinaten), flow [C, 2, H, W], Masken [C, H, W] bool.
        Rückgabe: d [M, 3] Verschiebung f0 -> f1, valid [M] bool, Residuum [M] px."""
        C, M = len(self.views), len(pts)
        cam, uv, z = self.project(pts)
        u, v = uv[..., 0].round().long(), uv[..., 1].round().long()
        inside = (z > 0) & (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
        uc, vc = u.clamp(0, self.W - 1), v.clamp(0, self.H - 1)
        idx = vc * self.W + uc
        # Sichtbarkeit: z-Buffer über die Hull-Punkte in Zellen von cell × cell Pixeln
        Wc, Hc = self.W // cell + 1, self.H // cell + 1
        cidx = (vc // cell) * Wc + (uc // cell)
        zbuf = torch.full((C, Hc * Wc), float("inf"), device=self.dev)
        zbuf.scatter_reduce_(1, cidx, torch.where(inside, z, torch.inf), reduce="amin")
        visible = inside & (z <= torch.gather(zbuf, 1, cidx) + 3 * voxel)
        fg0 = torch.gather(mask0.reshape(C, -1), 1, idx)
        fl = torch.stack([torch.gather(flow[:, k].reshape(C, -1), 1, idx) for k in range(2)], -1)  # [C, M, 2]
        u1 = (uv[..., 0] + fl[..., 0]).round().long().clamp(0, self.W - 1)
        v1 = (uv[..., 1] + fl[..., 1]).round().long().clamp(0, self.H - 1)
        fg1 = torch.gather(mask1.reshape(C, -1), 1, v1 * self.W + u1)
        ok = visible & fg0 & fg1  # [C, M]

        # Jacobi der Projektion nach Weltkoordinaten: [C, M, 2, 3]
        fx, fy = self.K[:, 0, 0][:, None], self.K[:, 1, 1][:, None]
        x, y = cam[..., 0], cam[..., 1]
        zc = z.clamp(min=1e-6)
        Jc = torch.zeros((C, M, 2, 3), device=self.dev)
        Jc[..., 0, 0] = fx / zc
        Jc[..., 0, 2] = -fx * x / zc**2
        Jc[..., 1, 1] = fy / zc
        Jc[..., 1, 2] = -fy * y / zc**2
        J = Jc @ self.w2c[:, None, :3, :3]  # [C, M, 2, 3]

        w = ok.float()
        d = torch.zeros((M, 3), device=self.dev)
        for _ in range(4):  # IRLS mit Huber-Gewichten
            Jw = J * w[..., None, None]
            A = (Jw.transpose(-1, -2) @ J).sum(0) + 1e-6 * torch.eye(3, device=self.dev)  # [M, 3, 3]
            b = (Jw.transpose(-1, -2) @ fl[..., None]).sum(0)  # [M, 3, 1]
            d = torch.linalg.solve(A, b)[..., 0]
            r = ((J @ d[None, :, :, None])[..., 0] - fl).norm(dim=-1)  # [C, M]
            w = ok.float() * torch.clamp(huber_px / r.clamp(min=1e-6), max=1.0)
        n_ok = ok.sum(0)
        valid = n_ok >= min_cams
        res = torch.where(ok, r, torch.nan).nanmedian(dim=0).values
        return d, valid, res, n_ok


def fill_invalid(pts, d, valid, k=8):
    """Ungültige Punkte: Mittel der k nächsten gültigen Nachbarn (räumlich)."""
    from sklearn.neighbors import NearestNeighbors

    if valid.all() or not valid.any():
        return d
    nn = NearestNeighbors(n_neighbors=min(k, int(valid.sum()))).fit(pts[valid])
    idx = nn.kneighbors(pts[~valid])[1]
    d = d.copy()
    d[~valid] = d[valid][idx].mean(1)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--exclude-cams", nargs="*", default=[], help="z. B. Testkameras")
    ap.add_argument("--scale", type=float, default=0.5, help="Auflösung für RAFT relativ zum Bild")
    ap.add_argument("--iters", type=int, default=12, help="RAFT-Iterationen")
    ap.add_argument("--min-cams", type=int, default=3)
    ap.add_argument("--start", type=int, default=0, help="erster Frame (Test)")
    ap.add_argument("--frames", type=int, default=-1, help="nur N Frames ab --start (Test)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    data = args.data_dir.expanduser()
    views = [v for v in load_colmap(data / "sparse" / "0") if v["name"] not in set(args.exclude_cams)]
    F = len(sorted((data / "points").glob("points3d_frame*.npy")))
    todo = range(args.start, F if args.frames <= 0 else min(F, args.start + args.frames))
    print(f"{len(views)} Kameras, {F} Frames, bearbeite {todo.start}..{todo.stop - 1}, RAFT bei {args.scale:.2f}x")
    solver = FlowSolver(views, args.device, args.scale, args.iters)
    dt = 1.0 / max(F - 1, 1)
    stats = []
    t0 = time.time()
    from sklearn.neighbors import NearestNeighbors

    for f in todo:
        f0, f1 = (f, f + 1) if f + 1 < F else (f, f - 1)  # letzter Frame: rückwärts
        pts_np = np.load(data / "points" / f"points3d_frame{f0:06d}.npy").astype(np.float32)
        pts = torch.from_numpy(pts_np).to(args.device)
        # Punktabstand des Hulls (für die z-Buffer-Toleranz): Median der Nächste-Nachbar-Distanz
        voxel = float(np.median(NearestNeighbors(n_neighbors=2).fit(pts_np).kneighbors(pts_np)[0][:, 1]))
        flow = solver.flows(data, f0, f1)
        m0 = torch.stack([torch.from_numpy(solver.read(data, "masks", v["name"], f0)) for v in views]).to(args.device) > 127
        m1 = torch.stack([torch.from_numpy(solver.read(data, "masks", v["name"], f1)) for v in views]).to(args.device) > 127
        d, valid, res, n_ok = solver.solve(pts, flow, m0, m1, voxel, args.min_cams)
        d, valid, res = d.cpu().numpy(), valid.cpu().numpy(), res.cpu().numpy()
        d = fill_invalid(pts_np, d, valid)
        vel = d / ((f1 - f0) * dt)  # Vorzeichen: rückwärts am Ende wird mitgedreht
        np.save(data / "points" / f"velocities_frame{f:06d}.npy", vel.astype(np.float32))
        zq = np.quantile(pts_np[:, 2], [0.1, 0.9])
        top, bot = pts_np[:, 2] >= zq[1], pts_np[:, 2] <= zq[0]
        st = dict(frame=f, valid_frac=float(valid.mean()), cams_median=float(np.median(n_ok.cpu().numpy())),
                  residual_px_median=float(np.nanmedian(res[valid])) if valid.any() else None,
                  speed_median=float(np.median(np.linalg.norm(vel, axis=1))),
                  dz_top_per_frame=float(np.mean(d[top, 2]) * np.sign(f1 - f0)),
                  dz_bottom_per_frame=float(np.mean(d[bot, 2]) * np.sign(f1 - f0)))
        stats.append(st)
        print(f"f{f:3d}: gültig {st['valid_frac']:.2f}, Kameras {st['cams_median']:.0f}, Residuum {st['residual_px_median']} px, "
              f"|v| {st['speed_median']:.3f}, dz/Frame oben {st['dz_top_per_frame']:+.4f} unten {st['dz_bottom_per_frame']:+.4f}  "
              f"[{time.time() - t0:.0f}s]", flush=True)
    with open(data / "points" / "velocities_stats.json", "w") as fh:
        json.dump(dict(args={k: str(v) for k, v in vars(args).items()}, frames=stats), fh, indent=2)


if __name__ == "__main__":
    main()
