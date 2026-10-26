# ma4dgs

4D Gaussian Splatting für dynamische Multi-View-Szenen, Masterarbeit. Eigene Implementierung der Methode aus dem Paper *FreeTimeGS* auf Basis von [gsplat](https://github.com/nerfstudio-project/gsplat) 1.6.0.

gsplat bleibt unverändert als Bibliothek in `~/masterarbeit/gsplat`; eigener Code liegt nur hier.

## Hinweise zur Herkunft

Alles, was von anderen stammt und hier verwendet wird:

**Methoden (Paper)**
- **FreeTimeGS**: Y. Wang et al., *FreeTimeGS: Free Gaussian Primitives at Anytime and Anywhere for Dynamic Scene Reconstruction*, CVPR 2025 ([Projektseite](https://zju3dv.github.io/freetimegs/)). Umgesetzt sind die dort beschriebene 4D-Repräsentation, die Losses, die 4D-Regularisierung und die Relokation (`ma4dgs/`, `train.py`).
- **FreeTimeGS++** (2026): Zeitplan und Schwelle der Relokation sowie die Vererbung der Parameter (`train.py`, `ma4dgs/relocation.py`).
- **3D Gaussian Splatting**: B. Kerbl et al., SIGGRAPH 2023. Grundrepräsentation, Lernraten, PLY-Format.
- **3D Gaussian Splatting as Markov Chain Monte Carlo**: S. Kheradmand et al., NeurIPS 2024. Prinzip, „tote“ Gaußsche zu verschieben und Opazität und Größe aufzuteilen (`ma4dgs/relocation.py`).

**Code**
- **gsplat (Apache-2.0)**, [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat) 1.6.0:
  - `train.py` ist an `gsplat/examples/simple_trainer.py` angelehnt, der Lizenzhinweis steht im Dateikopf.
  - `ma4dgs/relocation.py` ist an `gsplat/strategy/ops.py` (`relocate`) angelehnt.
  - Rasterisierer, SH-Auswertung und PLY-Export werden aus `gsplat` importiert, COLMAP-Parser und Hilfsfunktionen (`knn`, `rgb_to_sh`, `set_random_seed`) unverändert aus `gsplat/examples` (`datasets/`, `utils.py`).
- **FreeTimeGsVanilla (AGPL-3.0)**, [OpsiClear-4DGS/FreeTimeGsVanilla](https://github.com/OpsiClear-4DGS/FreeTimeGsVanilla): nur als Referenz ausgeführt und gelesen (`scripts/run_vanilla_ref.sh`), kein Code übernommen. Das Datenlayout (`images/camXX/`, `points/`) folgt seinem Format, damit beide dieselben Daten lesen.
- **BiRefNet (MIT)**, P. Zheng et al., *Bilateral Reference for High-Resolution Dichotomous Image Segmentation*, CAAI AIR 2024. Modell [`ZhengPeng7/BiRefNet`](https://huggingface.co/ZhengPeng7/BiRefNet), Revision `e2bf8e4` (dieselbe wie in 4DAnyone). `scripts/prepare_4danyone.py` lädt es über Hugging Face transformers (der Modellcode kommt per `trust_remote_code` aus dem Modell-Repository) und berechnet damit die Vordergrundmasken.

**Werkzeuge und Bibliotheken**
- COLMAP (Kameraposen, Triangulation), ffmpeg (Frames aus Videos)
- Hugging Face transformers, timm, kornia, einops (nur zum Laden und Ausführen von BiRefNet)
- PyTorch, torchmetrics (PSNR, SSIM, LPIPS mit den Netzgewichten von R. Zhang et al., CVPR 2018), viser und nerfview (Viewer), tyro, OpenCV, imageio, scikit-learn, pycolmap, NumPy, PyYAML, tqdm

**Methoden für generierte Ansichten** (`scripts/prepare_4danyone.py`)
- **Visual Hull / Space Carving**: A. Laurentini, *The Visual Hull Concept for Silhouette-Based Image Understanding*, IEEE TPAMI 1994. Eigene Umsetzung zur Initialisierung pro Frame.
- **3D-Skelett aus 4DAnyone** (Skelett-Prior, `ma4dgs/skeleton.py`): Die Punkte (`conditioning/keypoints_3d.npy`) erzeugt 4DAnyone, hier werden sie nur gelesen. Dahinter stehen **GVHMR** (Z. Shen et al., *World-Grounded Human Motion Recovery via Gravity-View Coordinates*, SIGGRAPH Asia 2024), das aus dem Eingabevideo eine **SMPL-X**-Sequenz schätzt (G. Pavlakos et al., *Expressive Body Capture: 3D Hands, Face, and Body from a Single Image*, CVPR 2019), und ein Regressor von 4DAnyone auf die 70 Punkte des **Goliath**-Schemas (R. Khirodkar et al., *Sapiens2*, ICLR 2026). Die Liste der Knochen in `ma4dgs/skeleton.py` folgt den Verbindungen, die 4DAnyone für sein Goliath40-Skelett zeichnet (`fdanyone/skeleton/keypoints.py`), ohne Fingerglieder, plus Hals–Nase und Augen-Querverbindung. Geschwindigkeitsfeld, Konfidenz und Regularisierung sind eigene Erweiterung.

**Daten**
- **Neural3DV / DyNeRF**: T. Li et al., *Neural 3D Video Synthesis from Multi-view Video*, CVPR 2022 (Meta). Szene `flame_steak`.
- **4DAnyone** (AntResearch, Apache-2.0), Y. Jin, T. Xie et al., *4DAnyone: Create Anyone in 4D from a Casual Monocular Video*, arXiv 2608.20335, 2026, [ant-research/4DAnyone](https://github.com/ant-research/4DAnyone): erzeugt aus einem eigenen Video synchrone Ansichten mit Kameras (`cameras.json`, `videos/dense/`). Ausgeführt auf einem anderen Rechner; hier wird nur seine Ausgabe gelesen.

**KI-Unterstützung**
- Code und Dokumentation sind mit Unterstützung von KI entstanden (Claude Code, Anthropic). Commits mit KI-Beteiligung tragen die Zeile `Co-Authored-By: Claude`.

## Umgebung

```bash
source ~/masterarbeit/.venv/bin/activate
cd ~/masterarbeit/ma4dgs     # Python nie in ~/masterarbeit selbst starten
```

## Verzeichnisse

| Pfad | Inhalt |
|---|---|
| `train.py` | eigener FreeTimeGS-Trainer (Konfiguration: `python train.py --help`) |
| `ma4dgs/` | Datensatz (`data.py`), 4D-Repräsentation (`model.py`), Relokation (`relocation.py`), Skelett-Prior (`skeleton.py`) |
| `scripts/` | Datenvorbereitung, Lauf-Skripte |
| `~/masterarbeit/data/n3dv/raw/<szene>/` | Original-Videos Neural3DV (`camXX.mp4`, `poses_bounds.npy`) |
| `~/masterarbeit/data/n3dv/<szene>_<N>f/` | vorbereitete Szene (siehe unten) |
| `~/masterarbeit/data/4danyone/<name>/` | vorbereitete 4DAnyone-Ausgabe (siehe „4DAnyone-Ausgabe vorbereiten“) |
| `~/masterarbeit/results/` | Trainingsläufe (TensorBoard, Checkpoints) |
| `EXPERIMENTS.md` | Protokoll aller Läufe |

## Training (eigener Trainer)

```bash
cd ~/masterarbeit/ma4dgs
bash scripts/run_proto.sh ~/masterarbeit/results/ftgs_proto_l1_kf1 \
    --data-dir ~/masterarbeit/data/n3dv/flame_steak_50f --keyframe-step 1
```

`run_proto.sh` ruft `train.py` auf und schreibt zusätzlich `run_info.txt` (Commit, Argumente, Wanduhrzeit, VRAM-Spitze laut `nvidia-smi`) und `vram.csv`.
Alle Bilder werden beim Start in den Hauptspeicher geladen (flame_steak, 50 Frames: 4 GB).
Testkamera ist `cam00` (`--test-cams`), ausgewertet werden alle Frames. Ergebnisse im Lauf-Ordner:

| Pfad | Inhalt |
|---|---|
| `stats/val_stepNNNNN.json` | Mittelwerte und PSNR/SSIM/LPIPS pro Frame |
| `stats/train_stepNNNNN.json` | Trainingszeit (ohne Evaluation), `torch`-Speicherspitze, Anzahl Gaußsche |
| `videos/test_cam00_stepN.mp4` | links GT, rechts Rendering, alle Frames |
| `videos/traj_stepN.mp4` | Kamerafahrt durch die Trainingskameras, Zeit läuft mit |
| `ckpts/ckpt_final.pt` | Parameter; nur auswerten mit `python train.py --ckpt <pfad> --result-dir <ordner>` |

Stand Schritt 4 (Machbarkeitsplan): feste Anzahl Gaußscher, Init aus den Punktwolken jedes `keyframe_step`-ten Frames (μₜ = Zeit des Frames, v = 0, s = 2 × Keyframe-Abstand), keine Relokation, keine 4D-Regularisierung.

## Neural3DV vorbereiten

```bash
# Download (ca. 1,2 GB pro Szene)
mkdir -p ~/masterarbeit/data/n3dv/raw && cd ~/masterarbeit/data/n3dv/raw
curl -L -O https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/flame_steak.zip
unzip flame_steak.zip

# Frames extrahieren (halbe Auflösung) + COLMAP auf Frame 0
cd ~/masterarbeit/ma4dgs
python scripts/prepare_n3dv.py \
    --src ~/masterarbeit/data/n3dv/raw/flame_steak \
    --out ~/masterarbeit/data/n3dv/flame_steak_50f \
    --num-frames 50 --downscale 2
```

Ergebnis:

```
flame_steak_50f/
  images/cam00/00000.png ... 00049.png   1352x1014, je Kamera ein Ordner
  sparse/0/                              COLMAP-Modell von Frame 0 (PINHOLE, Bildnamen camXX.png)
  frame0/                                nur Frame 0, direkt mit gsplat simple_trainer.py nutzbar
  colmap/                                Datenbank, Log, Eingabebilder; frames/NNNNN/sparse pro Frame
  points/                                Punktwolken pro Frame (triangulate_frames.py)
  poses_bounds.npy                       Original-LLFF-Posen (nur zum Vergleich)
  meta.json
```

Das Layout entspricht dem „flat“-Format, das auch FreeTimeGsVanilla liest (`--data-dir`, `--data-factor 1`).
`cam00` ist die Testkamera.

Punktwolke pro Frame (feste Posen aus `sparse/0`, SIFT + `colmap point_triangulator`, ca. 4 s pro Frame):

```bash
python scripts/triangulate_frames.py --scene ~/masterarbeit/data/n3dv/flame_steak_50f \
    --workers 8 --two-view-tracks
```

Das Ergebnis liegt in `points/points3d_frame000000.npy` und `points/colors_frame000000.npy` (`[M,3]` float32, Farben 0–255), wie FreeTimeGsVanilla es erwartet. Kennzahlen pro Frame stehen in `points/stats.json`.
Bei flame_steak sind es ca. 6700 Punkte pro Frame mit 0,55 px Reprojektionsfehler. `--two-view-tracks` behält auch Punkte, die nur zwei Kameras sehen, und verdoppelt damit etwa die Punktzahl.

Plausibilitätstest (statisch, Frame 0):

```bash
cd ~/masterarbeit/gsplat/examples
python simple_trainer.py mcmc --data_dir ~/masterarbeit/data/n3dv/flame_steak_50f/frame0 \
    --data_factor 1 --result_dir ~/masterarbeit/results/flame_steak_frame0 --max_steps 7000
```

## 4DAnyone-Ausgabe vorbereiten

Ausführliche Schritt-für-Schritt-Anleitung: `Anleitung_4DAnyone_zu_4DGS.md` im Workspace (Windows). Kurzform:

```bash
# einmalig: Pakete für BiRefNet
~/.local/bin/uv pip install --python ~/masterarbeit/.venv/bin/python transformers timm kornia einops

python scripts/prepare_4danyone.py --src /mnt/c/.../4DAnyone/<name> --out ~/masterarbeit/data/4danyone/<name>
bash scripts/run_4danyone.sh ~/masterarbeit/data/4danyone/<name> ~/masterarbeit/results/4danyone/<name>
python scripts/view_4d.py --ckpt ~/masterarbeit/results/4danyone/<name>/ckpts/ckpt_final.pt
```

`prepare_4danyone.py` schreibt dasselbe Layout wie `prepare_n3dv.py` und zusätzlich `masks/camXX/NNNNN.png`:
Frames (ffmpeg), Masken (BiRefNet), Punktwolke pro Frame als Visual Hull aus Masken und bekannten Kameras (statt SIFT-Triangulation), COLMAP-Modell direkt aus `cameras.json` (ohne COLMAP-Lauf).
Die Welt wird dabei von y-oben auf z-oben gedreht: Mit exakt +y als Oben-Richtung spiegelt `similarity_from_cameras` aus gsplat/examples die Szene, statt sie zu drehen.
Mit Masken lernt `train.py` nur den Vordergrund (`--background random|black|white`, `--lambda-mask`); `run_4danyone.sh` setzt das zusammen mit dem vollen Modell, `--test-cams` ohne Wert (alle Kameras trainieren) und `--traj-mode orbit`.

### Skelett-Prior (eigene Erweiterung, Branch `fast-motion-skeleton`)

4DAnyone rendert seine Skelett-Videos aus 3D-Punkten (`keypoints_3d.npy`), die es in einem versteckten Arbeitsordner neben der Ausgabe ablegt: `.<clip>.scratch-XXXXXXXX/conditioning/`. Er wird normalerweise am Ende gelöscht und bleibt nur stehen, wenn das Aufräumen scheitert. Liegt dieser `conditioning/`-Ordner in der 4DAnyone-Ausgabe (`--src`), übernimmt `prepare_4danyone.py` die Punkte nach `skeleton/` (Schritt 5; mit `--skip-frames --skip-masks --skip-hull` auch nachträglich).
`train.py --init-velocity skeleton` startet jede Gaußsche mit der Skelett-Geschwindigkeit an ihrer Position, `--lambda-skel` zieht die gelernte Geschwindigkeit während des ganzen Trainings dorthin (Abweichung in cm pro Frame, gewichtet mit der Nähe zum Skelett). `scripts/diag_skeleton.py <result_dir> ...` vergleicht gelernte und Skelett-Geschwindigkeit und zeigt PSNR/LPIPS getrennt nach ruhiger Phase und Sprung.

## Referenz: FreeTimeGsVanilla

[FreeTimeGsVanilla](https://github.com/OpsiClear-4DGS/FreeTimeGsVanilla) (AGPL-3.0) dient nur als Referenz, die ausgeführt und gelesen wird. **Von dort wird kein Code übernommen.**
Das Repo liegt außerhalb dieses Repositorys in `~/masterarbeit/refs/FreeTimeGsVanilla` (getestet mit Commit `42e2f9b`) und hat eine eigene `.venv` mit torch 2.10+cu128 und gsplat 1.5.3.

Installation (`uv sync --locked` scheitert, weil uv torch-scatter isoliert gegen ein torch mit CUDA 13 baut):

```bash
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
cd ~/masterarbeit/refs/FreeTimeGsVanilla
uv venv --python 3.12 --managed-python .venv && source .venv/bin/activate   # managed: bringt Python.h mit
uv export --locked --no-hashes --no-emit-project > /tmp/req.txt
uv pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install setuptools wheel ninja numpy==1.26.4
grep -vE '^(torch|torchvision|nvidia-|triton)==' /tmp/req.txt > /tmp/req2.txt
uv pip install --no-build-isolation -r /tmp/req2.txt                         # ca. 7 min
```

Lauf (Combiner, dann Training mit cam00 als einziger Testkamera, Auswertung aller Frames, VRAM-Protokoll):

```bash
cd ~/masterarbeit/ma4dgs
bash scripts/run_vanilla_ref.sh ~/masterarbeit/data/n3dv/flame_steak_50f \
    ~/masterarbeit/results/vanilla_flame_steak_50f_kf5 30000 5
```

Wichtige Standardwerte von Vanilla, die das Skript überschreibt: `test_every=8` (Kameras 0, 8, 16) und `eval_sample_every=60` (bei 50 Frames würde nur Frame 0 ausgewertet).
