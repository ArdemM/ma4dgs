"""Multi-View-Video im "flat"-Layout (siehe README, scripts/prepare_n3dv.py).

    <szene>/images/camXX/NNNNN.png   ein Ordner pro Kamera, ein Bild pro Frame
    <szene>/sparse/0                 COLMAP-Modell von Frame 0 (Bildnamen camXX.png)
    <szene>/frame0/                  images/camXX.png + sparse -> ../sparse
    <szene>/points/                  Punktwolken pro Frame (scripts/triangulate_frames.py)
    <szene>/masks/camXX/NNNNN.png    optional: Vordergrundmasken 0–255 (scripts/prepare_4danyone.py)
    <szene>/skeleton/                optional: 3D-Skelett pro Frame (scripts/prepare_4danyone.py)

Die Kameras stehen still, deshalb werden Posen und Intrinsics einmal mit dem
COLMAP-Parser aus gsplat/examples auf frame0/ gelesen (inkl. Normalisierung
des Weltkoordinatensystems). Die Punktwolken pro Frame liegen in den
ursprünglichen COLMAP-Koordinaten und werden mit derselben Transformation
normalisiert.
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch

GSPLAT_EXAMPLES = Path(
    os.environ.get("GSPLAT_EXAMPLES", "~/masterarbeit/gsplat/examples")
).expanduser()
if str(GSPLAT_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(GSPLAT_EXAMPLES))

from datasets.colmap import Parser  # noqa: E402
from datasets.normalize import transform_points  # noqa: E402


class MultiViewVideo:
    def __init__(
        self,
        scene_dir: str,
        test_cams: Sequence[str] = ("cam00",),
        num_frames: int = -1,
        normalize: bool = True,
        workers: int = 16,
    ):
        self.scene_dir = Path(scene_dir).expanduser()
        parser = Parser(
            str(self.scene_dir / "frame0"),
            factor=1,
            normalize=normalize,
            test_every=10**9,  # Split erfolgt hier über Kameranamen
        )
        for cid, params in parser.params_dict.items():
            assert len(params) == 0, f"Kamera {cid} hat Verzeichnung, erwartet PINHOLE"

        self.parser = parser
        self.cam_names = [Path(n).stem for n in parser.image_names]  # camXX
        self.camtoworlds = parser.camtoworlds.astype(np.float32)  # [C, 4, 4]
        self.Ks = np.stack([parser.Ks_dict[cid] for cid in parser.camera_ids]).astype(
            np.float32
        )  # [C, 3, 3]
        self.imsizes = [parser.imsize_dict[cid] for cid in parser.camera_ids]  # (W, H)
        self.transform = parser.transform
        self.scene_scale = float(parser.scene_scale)

        available = len(list((self.scene_dir / "images" / self.cam_names[0]).glob("*.png")))
        self.num_frames = available if num_frames <= 0 else min(num_frames, available)
        # Normierte Zeit t ∈ [0, 1], erster Frame 0, letzter Frame 1
        F = self.num_frames
        self.times = np.linspace(0.0, 1.0, F) if F > 1 else np.zeros(1)

        unknown = set(test_cams) - set(self.cam_names)
        assert not unknown, f"Unbekannte Testkameras: {unknown}"
        self.test_cams = [i for i, n in enumerate(self.cam_names) if n in test_cams]
        self.train_cams = [i for i, n in enumerate(self.cam_names) if n not in test_cams]
        self.train_items = [(c, f) for c in self.train_cams for f in range(F)]
        self.test_items = [(c, f) for c in self.test_cams for f in range(F)]

        self.images = self._load_images(workers)  # [C, F, H, W, 3] uint8
        # [C, F, H, W] uint8 oder None, falls der Datensatz keine Masken hat
        self.masks = self._load_images(workers, "masks") if (self.scene_dir / "masks").is_dir() else None

    def image_path(self, cam: int, frame: int, kind: str = "images") -> Path:
        return self.scene_dir / kind / self.cam_names[cam] / f"{frame:05d}.png"

    def _load_images(self, workers: int, kind: str = "images") -> torch.Tensor:
        """Alle Bilder (bzw. Masken) einmal dekodieren und im Hauptspeicher halten (uint8)."""
        W, H = self.imsizes[0]
        assert all(s == (W, H) for s in self.imsizes), "Kameras mit unterschiedlicher Auflösung"
        C, F = len(self.cam_names), self.num_frames
        gray = kind == "masks"
        images = torch.empty((C, F, H, W) if gray else (C, F, H, W, 3), dtype=torch.uint8)

        def load(item: Tuple[int, int]) -> None:
            c, f = item
            path = self.image_path(c, f, kind)
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
            assert img is not None, f"Bild fehlt: {path}"
            assert img.shape[:2] == (H, W), f"{path}: {img.shape}"
            images[c, f] = torch.from_numpy(img if gray else cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        items = [(c, f) for c in range(C) for f in range(F)]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(load, items))
        print(f"[Data] {kind}: {C} Kameras x {F} Frames geladen ({images.numel() / 1024**3:.1f} GB)")
        return images

    def frame_points(self, frame: int) -> Tuple[np.ndarray, np.ndarray]:
        """Punktwolke eines Frames im normalisierten Weltkoordinatensystem.

        Rückgabe: Punkte [M, 3] float32, Farben [M, 3] float32 in [0, 1].
        """
        pts = np.load(self.scene_dir / "points" / f"points3d_frame{frame:06d}.npy")
        rgb = np.load(self.scene_dir / "points" / f"colors_frame{frame:06d}.npy")
        pts = transform_points(self.transform, pts.astype(np.float64)).astype(np.float32)
        return pts, (rgb.astype(np.float32) / 255.0)

    @property
    def meter(self) -> float:
        """Länge von 1 m (bzw. 1 Einheit der COLMAP-Koordinaten) nach der Normalisierung."""
        return float(np.linalg.norm(self.transform[:3, 0]))

    def skeleton(self) -> Tuple[np.ndarray, List[str]]:
        """3D-Skelett im normalisierten Weltkoordinatensystem.

        Rückgabe: Punkte [F, K, 3] float32, Punktnamen [K] (Goliath, aus 4DAnyone).
        """
        d = self.scene_dir / "skeleton"
        assert (d / "keypoints3d.npy").exists(), (
            f"{d}/keypoints3d.npy fehlt: scripts/prepare_4danyone.py mit conditioning/ der 4DAnyone-Ausgabe"
        )
        kp = np.load(d / "keypoints3d.npy").astype(np.float64)[: self.num_frames]
        assert len(kp) == self.num_frames, f"Skelett hat {len(kp)} Frames, Szene {self.num_frames}"
        names = json.loads((d / "keypoint_names.json").read_text())
        F, K, _ = kp.shape
        kp = transform_points(self.transform, kp.reshape(-1, 3)).reshape(F, K, 3)
        return kp.astype(np.float32), names

    def keyframes(self, step: int) -> List[int]:
        return list(range(0, self.num_frames, step))
