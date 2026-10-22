# Minimaler FreeTimeGS-Trainer (Machbarkeitsplan Schritt 4).
#
# Aufbau (Optimizer, Lernraten, SH-Schedule, Loss, Evaluation) angelehnt an
# gsplat/examples/simple_trainer.py:
#   SPDX-FileCopyrightText: Copyright 2023-2026 the Regents of the University of
#   California, Nerfstudio Team and contributors. All rights reserved.
#   SPDX-License-Identifier: Apache-2.0
#
# Feste Anzahl Gaußscher (keine Densifizierung). Die FreeTimeGS-Bausteine aus
# Schritt 5 sind einzeln schaltbar und standardmäßig aus (= Prototyp aus Schritt 4):
#   --lambda-reg 1e-2            4D-Regularisierung (Gl. 6)
#   --relocate                   periodische Relokation (Gl. 7, ma4dgs/relocation.py)
#   --init-velocity knn          k-NN-Geschwindigkeit zwischen Keyframes
#   --velocities-lr-final 5e-5   Velocity-Annealing
#
# Aufruf:
#   python train.py --data-dir ~/masterarbeit/data/n3dv/flame_steak_50f \
#       --result-dir ~/masterarbeit/results/ftgs_proto

import json
import math
import os
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

import imageio
import numpy as np
import torch
import tqdm
import tyro
import yaml
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from ma4dgs.data import MultiViewVideo
from ma4dgs.model import gaussians_at, init_from_frames, knn_velocities, temporal_opacity
from ma4dgs.relocation import Relocator
from datasets.traj import focus_point_fn, viewmatrix  # gsplat/examples
from gsplat.losses import l1_loss, ssim_loss
from gsplat.rendering import rasterization
from utils import set_random_seed  # gsplat/examples/utils.py


@dataclass
class Config:
    # Vorbereitete Szene (scripts/prepare_n3dv.py + scripts/triangulate_frames.py)
    data_dir: str = "~/masterarbeit/data/n3dv/flame_steak_50f"
    result_dir: str = "~/masterarbeit/results/ftgs_proto"
    # Testkameras (werden nicht trainiert, alle Frames werden ausgewertet)
    test_cams: List[str] = field(default_factory=lambda: ["cam00"])
    # Anzahl Frames (-1 = alle vorhandenen)
    num_frames: int = -1
    # Checkpoint laden und nur auswerten
    ckpt: Optional[str] = None

    max_steps: int = 30_000
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    seed: int = 42

    # Initialisierung: Punktwolke jedes keyframe_step-ten Frames, μₜ = Zeit des Frames
    keyframe_step: int = 1
    # Anfangsdauer s in normierter Zeit; None = 2 × Keyframe-Abstand
    init_duration: Optional[float] = None
    # Untergrenze der Dauer s in Frames (0 = keine). Wird nach jedem Schritt
    # durch Klemmen von log s erzwungen (Vanilla: min 0.02 normiert ≈ 2,4
    # Frames). Ohne Untergrenze weicht das Modell bei schneller Bewegung in
    # „3DGS pro Frame“ aus (s ≈ 1,4 Frames, v → 0) statt v zu lernen.
    min_duration_frames: float = 0.0
    init_opacity: float = 0.1
    init_scale: float = 1.0
    sh_degree: int = 3
    sh_degree_interval: int = 1000

    # Lernraten (3DGS/gsplat-Werte; μₜ, s, v eigene Wahl, konstant)
    means_lr: float = 1.6e-4  # × scene_scale, exponentiell auf 1 % abfallend
    scales_lr: float = 5e-3
    quats_lr: float = 1e-3
    opacities_lr: float = 5e-2
    sh0_lr: float = 2.5e-3
    shN_lr: float = 2.5e-3 / 20
    times_lr: float = 1e-3
    durations_lr: float = 5e-3
    velocities_lr: float = 5e-3
    # Velocity-Annealing (Abschnitt 3.2): lr(p) = lr₀^(1−p) · lr₁^p, p = step / max_steps.
    # None = konstant velocities_lr. Endwert nennt das Paper nicht.
    velocities_lr_final: Optional[float] = None
    # Geschwindigkeit optimieren (False = statische Gaußsche, nur zeitlich begrenzt)
    learn_velocity: bool = True
    # Anfangsgeschwindigkeit: 0 oder k-NN-Zuordnung zum nächsten Keyframe
    # knn_centroid: kNN nach Ausgleich der Schwerpunktverschiebung (siehe knn_velocities)
    # file: points/velocities_frameNNNNNN.npy aus optischem Fluss (scripts/estimate_velocities.py)
    init_velocity: Literal["zero", "knn", "knn_centroid", "file"] = "zero"
    knn_k: int = 3

    # 4D-Regularisierung (Gl. 6): λ_reg · mean(σ · sg[σ(t)]), nur bis reg_until
    # ("frühe Trainingsphase", Länge im Paper nicht angegeben)
    lambda_reg: float = 0.0
    reg_until: int = 15_000

    # Periodische Relokation (Gl. 7), Zeitplan wie in FreeTimeGS++ (Suppl. A)
    relocate: bool = False
    reloc_every: int = 100
    reloc_start: int = 500
    reloc_stop: int = 25_000
    reloc_min_opacity: float = 5e-3
    reloc_lambda_grad: float = 0.5
    reloc_lambda_opa: float = 0.5
    # Gaußsche, die zeitlich ganz außerhalb der Sequenz liegen (nie gerendert,
    # kein Gradient), ebenfalls als tot behandeln und verschieben
    reloc_outside_dead: bool = False
    # Obergrenze des Anteils der pro Runde verschobenen Gaußschen (1.0 = alle toten)
    reloc_max_frac: float = 1.0
    # Zielwahl der Relokation zeitlich gewichten: Score × (Bildfehler des Frames
    # bei μₜ / mittlerer Bildfehler)^reloc_time_power. Der Fehler pro Frame ist
    # ein gleitender Mittelwert (EMA) des Bildlosses der Trainingsschritte.
    reloc_time_weight: bool = False
    reloc_time_power: float = 1.0
    reloc_err_ema: float = 0.1

    # Rendering-Loss (Gl. 5): λ_img·L_img + λ_ssim·(1 − SSIM) + λ_perc·LPIPS
    img_loss: Literal["l1", "l2"] = "l1"
    lambda_img: float = 0.8
    lambda_ssim: float = 0.2
    lambda_perc: float = 0.0

    # Hintergrund. "image" = Bild unverändert als Ziel (Neural3DV). Sonst braucht der
    # Datensatz masks/ (scripts/prepare_4danyone.py): Ziel = Bild·Maske + Hintergrund·(1 − Maske),
    # gerendert wird vor demselben Hintergrund. "random" = neue Zufallsfarbe pro Schritt
    # (Auswertung und Videos dann schwarz), zwingt transparente Bereiche auf Opazität 0.
    background: Literal["image", "black", "white", "random"] = "image"
    # L1 zwischen gerenderter Opazität und Maske (nur mit masks/)
    lambda_mask: float = 0.0

    # Gaußsche mit σ(t) unter diesem Wert werden nicht rasterisiert
    temporal_cull: float = 1e-3
    near_plane: float = 0.01
    far_plane: float = 1e10

    tb_every: int = 100
    lpips_net: Literal["vgg", "alex"] = "alex"
    # Trajektorien-Video (Kamerafahrt, Zeit läuft mit). "arc" = Bogen vor der Szene
    # (Kameras auf einer Seite, Neural3DV), "orbit" = voller Kreis (Kameras rundherum, 4DAnyone)
    render_traj: bool = True
    traj_mode: Literal["arc", "orbit"] = "arc"
    traj_frames: int = 300  # 10 s bei 30 FPS
    # Anteil der Kamera-Spannweite, den der Bogen abdeckt (< 1 bleibt im gut gesehenen Bereich)
    traj_arc_scale: float = 0.8
    load_workers: int = 16


class Runner:
    def __init__(self, cfg: Config) -> None:
        set_random_seed(cfg.seed)
        self.cfg = cfg
        self.device = "cuda"
        self.result_dir = Path(cfg.result_dir).expanduser()
        for sub in ["ckpts", "stats", "renders", "videos"]:
            (self.result_dir / sub).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.result_dir / "tb"))

        self.data = MultiViewVideo(
            cfg.data_dir, cfg.test_cams, cfg.num_frames, workers=cfg.load_workers
        )
        if cfg.background != "image" or cfg.lambda_mask > 0:
            assert self.data.masks is not None, f"--background/--lambda-mask brauchen {cfg.data_dir}/masks"
        self.scene_scale = self.data.scene_scale * 1.1
        F = self.data.num_frames
        print(
            f"Szene: {len(self.data.cam_names)} Kameras, {F} Frames, "
            f"Test: {[self.data.cam_names[c] for c in self.data.test_cams]}, "
            f"scene_scale {self.scene_scale:.3f}"
        )

        if cfg.init_duration is None:
            self.init_duration = 2.0 * cfg.keyframe_step / max(F - 1, 1)
        else:
            self.init_duration = cfg.init_duration
        keyframes = self.data.keyframes(cfg.keyframe_step)
        frames = [(*self.data.frame_points(f), self.data.times[f]) for f in keyframes]
        init = init_from_frames(
            frames,
            init_duration=self.init_duration,
            init_opacity=cfg.init_opacity,
            init_scale=cfg.init_scale,
            sh_degree=cfg.sh_degree,
            velocities=(
                [self.data.frame_velocities(f) for f in keyframes]
                if cfg.init_velocity == "file"
                else knn_velocities(frames, cfg.knn_k, align_centroid=cfg.init_velocity == "knn_centroid")
                if cfg.init_velocity != "zero"
                else None
            ),
        )
        self.splats = torch.nn.ParameterDict(
            {k: torch.nn.Parameter(v) for k, v in init.items()}
        ).to(self.device)
        self.splats["velocities"].requires_grad_(cfg.learn_velocity)
        print(
            f"Init: {len(keyframes)} Keyframes, {len(self.splats['means'])} Gaußsche, "
            f"Dauer s = {self.init_duration:.4f}, "
            f"|v| Median {self.splats['velocities'].norm(dim=-1).median().item():.4f}"
        )
        self.relocator = (
            Relocator(
                len(self.splats["means"]),
                self.device,
                lambda_grad=cfg.reloc_lambda_grad,
                lambda_opa=cfg.reloc_lambda_opa,
                min_opacity=cfg.reloc_min_opacity,
                time_range=(float(self.data.times[0]), float(self.data.times[-1]))
                if cfg.reloc_outside_dead
                else None,
                max_frac=cfg.reloc_max_frac,
            )
            if cfg.relocate
            else None
        )

        lrs = {
            "means": cfg.means_lr * self.scene_scale,
            "scales": cfg.scales_lr,
            "quats": cfg.quats_lr,
            "opacities": cfg.opacities_lr,
            "sh0": cfg.sh0_lr,
            "shN": cfg.shN_lr,
            "times": cfg.times_lr,
            "durations": cfg.durations_lr,
            "velocities": cfg.velocities_lr,
        }
        self.optimizers = {
            name: torch.optim.Adam(
                [{"params": self.splats[name], "lr": lr, "name": name}],
                eps=1e-15,
                fused=True,
            )
            for name, lr in lrs.items()
            if self.splats[name].requires_grad
        }

        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type=cfg.lpips_net, normalize=True
        ).to(self.device)

    def render(
        self,
        cam: int,
        t: float,
        sh_degree: int,
        c2w: Optional[torch.Tensor] = None,
        bg: Optional[torch.Tensor] = None,
    ):
        """Bild der Kamera cam (bzw. Pose c2w) zum Zeitpunkt t vor Hintergrund bg [1, 3], Form [1, H, W, 3]."""
        if c2w is None:
            c2w = torch.from_numpy(self.data.camtoworlds[cam]).to(self.device)
        K = torch.from_numpy(self.data.Ks[cam]).to(self.device)
        W, H = self.data.imsizes[cam]
        g, mask = gaussians_at(self.splats, t, self.cfg.temporal_cull)
        colors, alphas, info = rasterization(
            means=g["means"],
            quats=g["quats"],
            scales=g["scales"],
            opacities=g["opacities"],
            colors=g["colors"],
            viewmats=torch.linalg.inv(c2w)[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=sh_degree,
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
            backgrounds=bg,
        )
        return colors, alphas, info, mask

    def background(self, train: bool) -> Optional[torch.Tensor]:
        """Hintergrundfarbe [1, 3] oder None (= Bild unverändert, Rendern vor Schwarz)."""
        mode = self.cfg.background
        if mode == "image":
            return None
        if mode == "random" and train:
            return torch.rand((1, 3), device=self.device)
        return torch.full((1, 3), 1.0 if mode == "white" else 0.0, device=self.device)

    def target(self, cam: int, frame: int, bg: Optional[torch.Tensor]):
        """Zielbild [1, H, W, 3] in [0, 1] und Maske [1, H, W, 1] (None ohne masks/)."""
        pixels = self.data.images[cam, frame].to(self.device, non_blocking=True).float()[None] / 255.0
        if self.data.masks is None:
            return pixels, None
        m = self.data.masks[cam, frame].to(self.device, non_blocking=True).float()[None, ..., None] / 255.0
        if bg is not None:
            pixels = pixels * m + bg * (1.0 - m)
        return pixels, m

    def train(self) -> None:
        cfg = self.cfg
        with open(self.result_dir / "cfg.yml", "w") as f:
            yaml.dump(
                {**asdict(cfg), "init_duration_effective": self.init_duration,
                 "commit": git_commit()},
                f,
            )

        max_steps = cfg.max_steps
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
        )
        items = self.data.train_items
        rng = np.random.default_rng(cfg.seed)
        log_s_min = (
            math.log(cfg.min_duration_frames / max(self.data.num_frames - 1, 1))
            if cfg.min_duration_frames > 0
            else None
        )
        if log_s_min is not None:
            with torch.no_grad():
                self.splats["durations"].clamp_(min=log_s_min)
        # Bildfehler pro Frame (EMA), NaN = noch nicht gesehen
        frame_err = torch.full((self.data.num_frames,), float("nan"), device=self.device)

        torch.cuda.reset_peak_memory_stats()
        tic = time.time()
        visible_sum = 0
        pbar = tqdm.tqdm(range(max_steps), mininterval=10)
        for step in pbar:
            cam, frame = items[rng.integers(len(items))]
            t = float(self.data.times[frame])
            bg = self.background(train=True)
            pixels, fg = self.target(cam, frame, bg)  # [1, H, W, 3], [1, H, W, 1] / None

            if cfg.velocities_lr_final is not None and "velocities" in self.optimizers:
                p = step / max_steps
                lr = cfg.velocities_lr ** (1 - p) * cfg.velocities_lr_final**p
                self.optimizers["velocities"].param_groups[0]["lr"] = lr

            sh_degree = min(step // cfg.sh_degree_interval, cfg.sh_degree)
            colors, alphas, info, mask = self.render(cam, t, sh_degree, bg=bg)
            visible_sum += int(mask.sum())
            if self.relocator is not None:
                info["means2d"].retain_grad()

            if cfg.img_loss == "l1":
                imgloss = l1_loss(colors, pixels).mean()
            else:
                imgloss = torch.nn.functional.mse_loss(colors, pixels)
            ssimloss = ssim_loss(colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2))
            if cfg.reloc_time_weight:
                e = imgloss.detach()
                frame_err[frame] = e if torch.isnan(frame_err[frame]) else (1 - cfg.reloc_err_ema) * frame_err[frame] + cfg.reloc_err_ema * e
            loss = cfg.lambda_img * imgloss + cfg.lambda_ssim * ssimloss
            if cfg.lambda_perc > 0:
                lpipsloss = self.lpips(
                    colors.clamp(0, 1).permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2)
                )
                loss = loss + cfg.lambda_perc * lpipsloss
            maskloss = None
            if cfg.lambda_mask > 0:
                maskloss = (alphas - fg).abs().mean()
                loss = loss + cfg.lambda_mask * maskloss
            regloss = None
            if cfg.lambda_reg > 0 and step < cfg.reg_until:
                # Gl. 6: Basis-Opazität, gewichtet mit sg[σ(t)], gemittelt über alle N
                temp = temporal_opacity(self.splats, t).detach()
                regloss = (torch.sigmoid(self.splats["opacities"]) * temp).mean()
                loss = loss + cfg.lambda_reg * regloss
            loss.backward()

            if self.relocator is not None:
                self.relocator.accumulate(info, mask)

            for opt in self.optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)
            scheduler.step()
            if log_s_min is not None:
                with torch.no_grad():
                    self.splats["durations"].clamp_(min=log_s_min)

            if (
                self.relocator is not None
                and cfg.reloc_start <= step + 1 <= cfg.reloc_stop
                and (step + 1) % cfg.reloc_every == 0
            ):
                target_weight = None
                if cfg.reloc_time_weight:
                    err = torch.nan_to_num(frame_err, nan=float(torch.nanmean(frame_err)))
                    err = err / err.mean()
                    fr = (self.splats["times"].detach().squeeze(-1).clamp(0, 1) * (self.data.num_frames - 1)).round().long()
                    target_weight = err[fr] ** cfg.reloc_time_power
                    self.writer.add_scalar("reloc/frame_err_max_over_min", (err.max() / err.min()).item(), step)
                n_reloc = self.relocator.step(self.splats, self.optimizers, target_weight=target_weight)
                self.writer.add_scalar("gauss/relocated", n_reloc, step)
                self.writer.add_scalar("gauss/relocated_outside_time", self.relocator.n_dead_time, step)

            if step % cfg.tb_every == 0:
                pbar.set_description(f"loss={loss.item():.3f} sh={sh_degree} sichtbar={int(mask.sum())}")
                w = self.writer
                w.add_scalar("train/loss", loss.item(), step)
                w.add_scalar("train/imgloss", imgloss.item(), step)
                w.add_scalar("train/ssimloss", ssimloss.item(), step)
                if regloss is not None:
                    w.add_scalar("train/regloss", regloss.item(), step)
                if maskloss is not None:
                    w.add_scalar("train/maskloss", maskloss.item(), step)
                if "velocities" in self.optimizers:
                    w.add_scalar("train/velocities_lr", self.optimizers["velocities"].param_groups[0]["lr"], step)
                w.add_scalar("train/visible_GS", int(mask.sum()), step)
                w.add_scalar("train/mem_GB", torch.cuda.max_memory_allocated() / 1024**3, step)
                with torch.no_grad():
                    s = torch.exp(self.splats["durations"])
                    w.add_scalar("gauss/duration_median", s.median().item(), step)
                    opa = torch.sigmoid(self.splats["opacities"])
                    w.add_scalar("gauss/opacity_mean", opa.mean().item(), step)
                    w.add_scalar("gauss/opacity_gt_0.9", (opa > 0.9).float().mean().item(), step)
                    w.add_scalar("gauss/dead_frac", (opa <= cfg.reloc_min_opacity).float().mean().item(), step)
                    w.add_scalar("gauss/speed_median", self.splats["velocities"].norm(dim=-1).median().item(), step)

            if step + 1 in cfg.eval_steps or step + 1 == max_steps:
                train_time = time.time() - tic
                stats = {
                    "step": step + 1,
                    "train_time_s": train_time,
                    "mem_peak_GB": torch.cuda.max_memory_allocated() / 1024**3,
                    "num_GS": len(self.splats["means"]),
                    "visible_GS_mean": visible_sum / (step + 1),
                }
                print(stats)
                with open(self.result_dir / "stats" / f"train_step{step + 1:05d}.json", "w") as f:
                    json.dump(stats, f, indent=2)
                self.eval(step + 1)
                tic = time.time() - train_time  # Evaluationszeit nicht mitzählen

        torch.save(
            {"step": max_steps, "splats": self.splats.state_dict(), "cfg": asdict(cfg)},
            self.result_dir / "ckpts" / "ckpt_final.pt",
        )
        if cfg.render_traj:
            self.render_traj(max_steps)

    @torch.no_grad()
    def eval(self, step: int) -> dict:
        cfg = self.cfg
        if not self.data.test_cams:
            print(f"Evaluation (Schritt {step}) übersprungen: keine Testkamera")
            return {}
        print(f"Evaluation (Schritt {step}) ...")
        metrics = defaultdict(list)
        per_frame = []
        render_time = 0.0
        bg = self.background(train=False)
        for cam in self.data.test_cams:
            name = self.data.cam_names[cam]
            writer = imageio.get_writer(
                self.result_dir / "videos" / f"test_{name}_step{step}.mp4", fps=30, macro_block_size=2
            )
            for frame in range(self.data.num_frames):
                t = float(self.data.times[frame])
                pixels, _ = self.target(cam, frame, bg)
                torch.cuda.synchronize()
                tic = time.time()
                colors, _, _, mask = self.render(cam, t, cfg.sh_degree, bg=bg)
                torch.cuda.synchronize()
                render_time += time.time() - tic
                colors = colors.clamp(0.0, 1.0)

                p = pixels.permute(0, 3, 1, 2)
                c = colors.permute(0, 3, 1, 2)
                m = {
                    "psnr": self.psnr(c, p).item(),
                    "ssim": self.ssim(c, p).item(),
                    "lpips": self.lpips(c, p).item(),
                }
                for k, v in m.items():
                    metrics[k].append(v)
                per_frame.append({"cam": name, "frame": frame, "visible_GS": int(mask.sum()), **m})

                canvas = torch.cat([pixels, colors], dim=2)[0].cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)
                writer.append_data(canvas)
                if frame in (0, self.data.num_frames // 2, self.data.num_frames - 1):
                    imageio.imwrite(
                        self.result_dir / "renders" / f"test_{name}_f{frame:05d}_step{step}.png",
                        canvas,
                    )
            writer.close()

        n = len(per_frame)
        stats = {k: float(np.mean(v)) for k, v in metrics.items()}
        stats.update(
            {
                "step": step,
                "num_images": n,
                "render_fps": n / render_time,
                "num_GS": len(self.splats["means"]),
            }
        )
        print(
            f"PSNR {stats['psnr']:.2f} | SSIM {stats['ssim']:.4f} | LPIPS {stats['lpips']:.4f} "
            f"| {stats['render_fps']:.0f} FPS | {n} Bilder"
        )
        # Aufschlüsselung pro Testkamera (Kameras in anderer Reihe prüfen neue Blickwinkel)
        per_cam = {}
        for name in sorted({r["cam"] for r in per_frame}):
            rows = [r for r in per_frame if r["cam"] == name]
            per_cam[name] = {k: float(np.mean([r[k] for r in rows])) for k in ["psnr", "ssim", "lpips"]}
            if len(self.data.test_cams) > 1:
                c = per_cam[name]
                print(f"  {name}: PSNR {c['psnr']:.2f} | SSIM {c['ssim']:.4f} | LPIPS {c['lpips']:.4f}")
        with open(self.result_dir / "stats" / f"val_step{step:05d}.json", "w") as f:
            json.dump({"mean": stats, "per_cam": per_cam, "per_frame": per_frame}, f, indent=2)
        for k in ["psnr", "ssim", "lpips", "render_fps"]:
            self.writer.add_scalar(f"val/{k}", stats[k], step)
        self.writer.flush()
        return stats

    @torch.no_grad()
    def render_traj(self, step: int) -> None:
        """Kamerafahrt um den Fokuspunkt, die Zeit läuft dabei von 0 bis 1.

        "arc": Die Neural3DV-Kameras hängen in zwei Reihen übereinander. Eine Fahrt *durch*
        die Kameras springt deshalb ständig zwischen den Reihen. Stattdessen schwenkt
        die Kamera auf mittlerer Höhe sinusförmig hin und her (Start und Ende in der
        Mitte, ohne Ruck) und blickt immer auf den Punkt, auf den alle Kameras zielen.

        "orbit": Bei Kameras rundherum (4DAnyone) liegt die mittlere Kameraposition im
        Zentrum. Die Fahrt startet bei der ersten Trainingskamera und umrundet die Szene
        einmal auf mittlerem Abstand und mittlerer Höhe der Kameras.
        """
        cfg = self.cfg
        c2ws = self.data.camtoworlds[self.data.train_cams]
        focus = focus_point_fn(c2ws)
        # Kamera-y zeigt nach unten (OpenCV), die Welt-Oben-Richtung ist das Negative
        up = -c2ws[:, :3, 1].mean(0)
        up /= np.linalg.norm(up)
        rel = c2ws[:, :3, 3] - focus
        n = cfg.traj_frames

        if cfg.traj_mode == "orbit":
            heights = rel @ up
            radial = rel - heights[:, None] * up
            ref = radial[0] / np.linalg.norm(radial[0])
            offset = ref * np.linalg.norm(radial, axis=1).mean() + up * heights.mean()
            angles = 2 * np.pi * np.arange(n) / n
            print(f"Kamerafahrt: Kreis, {n} Bilder")
        else:
            offset = c2ws[:, :3, 3].mean(0) - focus  # mittlere Kameraposition relativ zum Fokus
            # Halbe Bogenweite aus der Verteilung der Kameras (Azimut um die Oben-Achse)
            ref = offset - (offset @ up) * up
            ref /= np.linalg.norm(ref)
            side = np.cross(up, ref)
            azimuths = np.arctan2(rel @ side, rel @ ref)
            half = cfg.traj_arc_scale * 0.5 * (azimuths.max() - azimuths.min())
            print(f"Kamerafahrt: Bogen ±{np.degrees(half):.1f}°, {n} Bilder")
            angles = half * np.sin(2 * np.pi * np.arange(n) / n)
        path = np.zeros((n, 4, 4), np.float32)
        for i, a in enumerate(angles):
            # Rodrigues-Drehung von offset um die Oben-Achse
            o = offset * np.cos(a) + np.cross(up, offset) * np.sin(a) + up * (up @ offset) * (1 - np.cos(a))
            pos = focus + o
            path[i, :3] = viewmatrix(focus - pos, -up, pos)
            path[i, 3, 3] = 1.0
        times = np.linspace(0.0, 1.0, n)
        cams = self.data.train_cams

        bg = self.background(train=False)
        writer = imageio.get_writer(self.result_dir / "videos" / f"traj_step{step}.mp4", fps=30, macro_block_size=2)
        for i in tqdm.trange(len(path), desc="Trajektorie"):
            c2w = torch.from_numpy(path[i]).to(self.device)
            colors, _, _, _ = self.render(cams[0], float(times[i]), self.cfg.sh_degree, c2w=c2w, bg=bg)
            writer.append_data((colors[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
        writer.close()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent, text=True
        ).strip()
    except Exception:
        return "unknown"


def main(cfg: Config) -> None:
    runner = Runner(cfg)
    if cfg.ckpt is not None:
        ckpt = torch.load(os.path.expanduser(cfg.ckpt), map_location=runner.device)
        for k in runner.splats.keys():
            runner.splats[k].data = ckpt["splats"][k]
        runner.eval(ckpt["step"])
        if cfg.render_traj:
            runner.render_traj(ckpt["step"])
        return
    runner.train()


if __name__ == "__main__":
    main(tyro.cli(Config))
