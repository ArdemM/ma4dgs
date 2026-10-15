"""Prepare a 4DAnyone output folder for 4D Gaussian Splatting with train.py.

4DAnyone (AntResearch) turns a monocular video into synchronized videos of N
virtual cameras with known poses. This script converts its output into the
"flat" layout of prepare_n3dv.py, so MultiViewVideo, train.py, view_4d.py and
export_ply.py work unchanged:

  1. Frames: videos/dense/XX.mp4 -> images/camXX/NNNNN.png (ffmpeg)
  2. Masks: foreground per image with BiRefNet (the segmentation model 4DAnyone
     itself uses) -> masks/camXX/NNNNN.png, 0-255. The generated backgrounds
     differ between views, so train.py should only learn the person.
  3. Points: visual hull per frame from masks + known cameras, surface voxels
     colored from the views that see them -> points/ (format of triangulate_frames.py).
     SIFT triangulation is unreliable on few, wide-baseline, generated views.
  4. COLMAP model: sparse/0 (PINHOLE, poses from cameras.json, frame-0 hull as
     points3D) and frame0/, without running COLMAP.

Input (4DAnyone output):
    <src>/cameras.json            K, camera_to_world (OpenCV), image size per camera
    <src>/videos/dense/XX.mp4     one video per camera_id
    <src>/metadata.json           (copied to meta.json for reference)

Output:
    <out>/images/camXX/NNNNN.png, masks/camXX/NNNNN.png, points/*.npy,
    sparse/0/{cameras,images,points3D}.txt, frame0/, meta.json

Example:
    python scripts/prepare_4danyone.py \
        --src /mnt/c/Users/Mirevi/Desktop/Masterarbeit_AM/4DAnyone/4DAnyone \
        --out ~/masterarbeit/data/4danyone/jumprope_6v
"""

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as Fn

from prepare_n3dv import probe, run

BIREFNET_REPO = "ZhengPeng7/BiRefNet"
BIREFNET_REVISION = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"  # as in 4DAnyone's metadata.json


# 4DAnyone world (y up, z front) -> z up, y back. With exactly +y up, the normalization in
# gsplat/examples (similarity_from_cameras, case c == -1) applies a reflection instead of a
# rotation, which leaves the cameras with det(R) = -1. Points and cameras both use this frame.
Y_UP_TO_Z_UP = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)


def load_cameras(src: Path) -> list[dict]:
    meta = json.loads((src / "cameras.json").read_text())
    # cameras.json says OPENCV but lists no distortion coefficients, i.e. pinhole
    assert meta["camera_model"] in ("PINHOLE", "OPENCV"), meta["camera_model"]
    assert meta["world_frame"]["axes"]["y"] == "up", meta["world_frame"]
    cams = []
    for c in sorted(meta["cameras"], key=lambda c: c["camera_id"]):
        cams.append({
            "name": f"cam{c['camera_id']:02d}",
            "video": src / c["video"],
            "K": np.array(c["K"], dtype=np.float64),
            "c2w": Y_UP_TO_Z_UP @ np.array(c["camera_to_world"], dtype=np.float64),
            "W": int(c["image_width"]),
            "H": int(c["image_height"]),
        })
    return cams


# --- 1. Frames ---------------------------------------------------------------------------------

def extract_frames(video: Path, out_dir: Path, num_frames: int) -> int:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-start_number", "0"]
    if num_frames > 0:
        cmd += ["-frames:v", str(num_frames)]
    run(cmd + [str(out_dir / "%05d.png")])
    return len(list(out_dir.glob("*.png")))


# --- 2. Masks ----------------------------------------------------------------------------------

@torch.no_grad()
def compute_masks(out: Path, cams: list[dict], num_frames: int, batch: int) -> None:
    from transformers import AutoModelForImageSegmentation

    model = AutoModelForImageSegmentation.from_pretrained(
        BIREFNET_REPO, trust_remote_code=True, revision=BIREFNET_REVISION
    ).cuda().eval().half()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

    items = [(c["name"], f) for c in cams for f in range(num_frames)]
    for c in cams:
        (out / "masks" / c["name"]).mkdir(parents=True, exist_ok=True)

    def read(item):
        img = cv2.imread(str(out / "images" / item[0] / f"{item[1]:05d}.png"), cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with ThreadPoolExecutor(8) as pool:
        for i in range(0, len(items), batch):
            chunk = items[i:i + batch]
            imgs = list(pool.map(read, chunk))
            H, W = imgs[0].shape[:2]
            x = torch.from_numpy(np.stack(imgs)).cuda().permute(0, 3, 1, 2).float() / 255.0
            x = Fn.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False)
            pred = model(((x - mean) / std).half())[-1].sigmoid().float()
            pred = Fn.interpolate(pred, size=(H, W), mode="bilinear", align_corners=False)
            pred = (pred[:, 0].clamp(0, 1) * 255).round().byte().cpu().numpy()
            for (name, f), m in zip(chunk, pred):
                cv2.imwrite(str(out / "masks" / name / f"{f:05d}.png"), m)
            if (i // batch) % 50 == 0:
                print(f"  masks {i + len(chunk)}/{len(items)}", flush=True)


# --- 3. Visual hull ----------------------------------------------------------------------------

class Hull:
    """Space carving with known cameras on the GPU."""

    def __init__(self, cams: list[dict], mask_thresh: float, tolerance: int):
        dev = "cuda"
        self.cams = cams
        self.w2c = torch.tensor(np.stack([np.linalg.inv(c["c2w"]) for c in cams]), dtype=torch.float32, device=dev)
        self.K = torch.tensor(np.stack([c["K"] for c in cams]), dtype=torch.float32, device=dev)
        self.W, self.H = cams[0]["W"], cams[0]["H"]
        self.thresh = mask_thresh
        self.min_votes = len(cams) - tolerance
        # Search box: cube around the point all cameras look at, half size 0.7 × nearest camera distance
        c2w = np.stack([c["c2w"] for c in cams])
        origins, dirs = c2w[:, :3, 3], c2w[:, :3, 2]
        A = sum(np.eye(3) - np.outer(d, d) for d in dirs)
        b = sum((np.eye(3) - np.outer(d, d)) @ o for o, d in zip(origins, dirs))
        focus = np.linalg.solve(A, b)
        half = 0.7 * np.linalg.norm(origins - focus, axis=1).min()
        self.box = (focus - half, focus + half)

    def project(self, pts: torch.Tensor):
        """pts [M, 3] -> pixel coords [C, M, 2] (float), depth [C, M]."""
        cam = pts @ self.w2c[:, :3, :3].transpose(1, 2) + self.w2c[:, None, :3, 3]
        z = cam[..., 2]
        uv = (cam @ self.K.transpose(1, 2))[..., :2] / z.clamp(min=1e-6)[..., None]
        return uv, z

    def carve(self, masks: torch.Tensor, lo: np.ndarray, hi: np.ndarray, res: int):
        """masks [C, H, W] float 0-1. Returns occupancy [X, Y, Z] bool, grid origin, voxel size."""
        voxel = float((hi - lo).max()) / res
        dims = np.maximum(np.ceil((hi - lo) / voxel).astype(int), 1)
        axes = [torch.arange(n, device="cuda", dtype=torch.float32) * voxel + float(l) + voxel / 2
                for n, l in zip(dims, lo)]
        occ = torch.empty(tuple(dims), dtype=torch.bool, device="cuda")
        for i in range(dims[0]):  # slice by slice to bound memory
            gy, gz = torch.meshgrid(axes[1], axes[2], indexing="ij")
            pts = torch.stack([torch.full_like(gy, axes[0][i]), gy, gz], dim=-1).reshape(-1, 3)
            uv, z = self.project(pts)
            u, v = uv[..., 0].round().long(), uv[..., 1].round().long()
            inside = (z > 0) & (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
            idx = v.clamp(0, self.H - 1) * self.W + u.clamp(0, self.W - 1)
            fg = torch.gather(masks.reshape(len(masks), -1), 1, idx) >= self.thresh
            occ[i] = ((inside & fg).sum(0) >= self.min_votes).reshape(dims[1], dims[2])
        return occ, lo, voxel

    def frame_points(self, images: torch.Tensor, masks: torch.Tensor, res: int, max_points: int,
                     rng: np.random.Generator):
        """images [C, H, W, 3] uint8, masks [C, H, W] float. Returns points [M, 3], colors [M, 3] 0-255."""
        lo, hi = self.box
        # Coarse pass in the whole box, fine pass in the bounding box of the coarse hull
        occ, origin, voxel = self.carve(masks, lo, hi, 96)
        nz = occ.nonzero()
        if len(nz) == 0:
            raise RuntimeError("visual hull is empty, check masks / cameras")
        lo = origin + (nz.min(0).values.cpu().numpy() - 2) * voxel
        hi = origin + (nz.max(0).values.cpu().numpy() + 3) * voxel
        occ, origin, voxel = self.carve(masks, lo, hi, res)

        # Surface voxels: occupied with at least one empty 6-neighbour
        empty = (~occ).float()[None, None]
        k = torch.zeros((1, 1, 3, 3, 3), device="cuda")
        k[0, 0, 1, 1, :] = k[0, 0, 1, :, 1] = k[0, 0, :, 1, 1] = 1
        surface = occ & (Fn.conv3d(Fn.pad(empty, (1,) * 6, value=1.0), k)[0, 0] > 0)
        ijk = surface.nonzero()
        pts = torch.from_numpy(origin).float().cuda() + (ijk.float() + 0.5) * voxel

        # Colour from the cameras that see the voxel (z-buffer over surface voxels)
        uv, z = self.project(pts)
        u, v = uv[..., 0].round().long(), uv[..., 1].round().long()
        inside = (z > 0) & (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
        idx = v.clamp(0, self.H - 1) * self.W + u.clamp(0, self.W - 1)
        zbuf = torch.full((len(self.cams), self.H * self.W), float("inf"), device="cuda")
        zbuf.scatter_reduce_(1, idx, torch.where(inside, z, torch.inf), reduce="amin")
        visible = inside & (z <= torch.gather(zbuf, 1, idx) + 2 * voxel)
        rgb = torch.gather(images.reshape(len(self.cams), -1, 3).float(), 1, idx[..., None].expand(-1, -1, 3))
        w = visible.float()
        w = torch.where(w.sum(0, keepdim=True) > 0, w, inside.float())  # unseen: all cameras
        colors = (rgb * w[..., None]).sum(0) / w.sum(0).clamp(min=1)[..., None]

        pts, colors = pts.cpu().numpy(), colors.cpu().numpy()
        if len(pts) > max_points:
            keep = rng.choice(len(pts), max_points, replace=False)
            pts, colors = pts[keep], colors[keep]
        pts = pts + rng.uniform(-0.5, 0.5, pts.shape) * voxel  # break up the grid
        return pts.astype(np.float32), colors.astype(np.float32), voxel


def build_points(out: Path, cams: list[dict], num_frames: int, args) -> None:
    hull = Hull(cams, args.mask_thresh, args.hull_tolerance)
    (out / "points").mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for f in range(num_frames):
        imgs, msks = [], []
        for c in cams:
            img = cv2.imread(str(out / "images" / c["name"] / f"{f:05d}.png"), cv2.IMREAD_COLOR)
            imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            msks.append(cv2.imread(str(out / "masks" / c["name"] / f"{f:05d}.png"), cv2.IMREAD_GRAYSCALE))
        images = torch.from_numpy(np.stack(imgs)).cuda()
        masks = torch.from_numpy(np.stack(msks)).cuda().float() / 255.0
        pts, rgb, voxel = hull.frame_points(images, masks, args.hull_res, args.points_per_frame, rng)
        np.save(out / "points" / f"points3d_frame{f:06d}.npy", pts)
        np.save(out / "points" / f"colors_frame{f:06d}.npy", rgb)
        if f % 20 == 0 or f == num_frames - 1:
            print(f"  hull frame {f}: {len(pts)} points, voxel {voxel * 100:.2f} cm", flush=True)


# --- 4. COLMAP model ---------------------------------------------------------------------------

def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> COLMAP quaternion (w, x, y, z)."""
    K = np.array([
        [R[0, 0] - R[1, 1] - R[2, 2], 0, 0, 0],
        [R[1, 0] + R[0, 1], R[1, 1] - R[0, 0] - R[2, 2], 0, 0],
        [R[2, 0] + R[0, 2], R[2, 1] + R[1, 2], R[2, 2] - R[0, 0] - R[1, 1], 0],
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1], R[0, 0] + R[1, 1] + R[2, 2]],
    ]) / 3.0
    vals, vecs = np.linalg.eigh(K)
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    return -q if q[0] < 0 else q


def write_colmap(out: Path, cams: list[dict]) -> None:
    sparse = out / "sparse" / "0"
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir(parents=True)
    with open(sparse / "cameras.txt", "w") as f:
        for i, c in enumerate(cams, start=1):
            K = c["K"]
            params = " ".join(repr(float(x)) for x in (K[0, 0], K[1, 1], K[0, 2], K[1, 2]))
            f.write(f"{i} PINHOLE {c['W']} {c['H']} {params}\n")
    with open(sparse / "images.txt", "w") as f:
        for i, c in enumerate(cams, start=1):
            w2c = np.linalg.inv(c["c2w"])
            q, t = rotmat_to_qvec(w2c[:3, :3]), w2c[:3, 3]
            f.write(f"{i} {' '.join(repr(float(x)) for x in q)} {' '.join(repr(float(x)) for x in t)} "
                    f"{i} {c['name']}.png\n\n")
    pts = np.load(out / "points" / "points3d_frame000000.npy")
    rgb = np.load(out / "points" / "colors_frame000000.npy").round().astype(int)
    with open(sparse / "points3D.txt", "w") as f:
        for i, (p, c) in enumerate(zip(pts, rgb), start=1):
            f.write(f"{i} {' '.join(repr(float(x)) for x in p)} {c[0]} {c[1]} {c[2]} 0\n")


def make_frame0(out: Path, cams: list[dict]) -> None:
    frame0 = out / "frame0"
    if frame0.exists():
        shutil.rmtree(frame0)
    (frame0 / "images").mkdir(parents=True)
    for c in cams:
        shutil.copy2(out / "images" / c["name"] / "00000.png", frame0 / "images" / f"{c['name']}.png")
    (frame0 / "sparse").symlink_to("../sparse")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="4DAnyone output folder (with cameras.json)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num-frames", type=int, default=-1, help="-1 = all frames of the videos")
    ap.add_argument("--skip-frames", action="store_true", help="reuse extracted frames")
    ap.add_argument("--skip-masks", action="store_true", help="reuse computed masks")
    ap.add_argument("--skip-hull", action="store_true", help="reuse computed points/")
    ap.add_argument("--mask-batch", type=int, default=4)
    ap.add_argument("--mask-thresh", type=float, default=0.5, help="mask value counted as foreground")
    ap.add_argument("--hull-res", type=int, default=256, help="voxels along the longest side of the person")
    ap.add_argument("--hull-tolerance", type=int, default=0,
                    help="number of cameras that may disagree before a voxel is carved away")
    ap.add_argument("--points-per-frame", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    src, out = args.src.expanduser(), args.out.expanduser()
    cams = load_cameras(src)
    sizes = {(c["W"], c["H"]) for c in cams}
    assert len(sizes) == 1, f"cameras with different image sizes: {sizes}"
    info = probe(cams[0]["video"])
    assert (int(info["width"]), int(info["height"])) == (cams[0]["W"], cams[0]["H"]), info
    print(f"{len(cams)} cameras, {cams[0]['W']}x{cams[0]['H']}, fps {info['r_frame_rate']}")
    out.mkdir(parents=True, exist_ok=True)

    if not args.skip_frames:
        print("[1/4] frames")
        with ThreadPoolExecutor(args.workers) as pool:
            counts = list(pool.map(lambda c: extract_frames(c["video"], out / "images" / c["name"], args.num_frames), cams))
        if len(set(counts)) != 1:
            print(f"WARNING: videos have different lengths {counts}, using the shortest")
    num_frames = min(len(list((out / "images" / c["name"]).glob("*.png"))) for c in cams)

    if not args.skip_masks:
        print(f"[2/4] masks (BiRefNet, {len(cams) * num_frames} images)")
        compute_masks(out, cams, num_frames, args.mask_batch)

    if not args.skip_hull:
        print(f"[3/4] visual hull, {num_frames} frames")
        build_points(out, cams, num_frames, args)

    print("[4/4] COLMAP model + frame0")
    write_colmap(out, cams)
    make_frame0(out, cams)

    meta = {
        "source": str(src),
        "cameras": [c["name"] for c in cams],
        "num_frames": num_frames,
        "fps": info["r_frame_rate"],
        "resolution": [cams[0]["W"], cams[0]["H"]],
        "masks": f"{BIREFNET_REPO}@{BIREFNET_REVISION}",
        "hull": {"res": args.hull_res, "tolerance": args.hull_tolerance, "mask_thresh": args.mask_thresh,
                 "points_per_frame": args.points_per_frame},
    }
    if (src / "metadata.json").exists():
        meta["4danyone"] = json.loads((src / "metadata.json").read_text())
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done: {out}")


if __name__ == "__main__":
    main()
