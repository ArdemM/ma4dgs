#!/bin/bash
# 4D-Rekonstruktion einer mit scripts/prepare_4danyone.py vorbereiteten Szene.
#
# Volles Modell wie ablation/kf1_full (Regularisierung, Relokation, kNN-Velocity,
# Annealing, alle Frames als Keyframes), dazu für die generierten Ansichten:
#   --background random --lambda-mask 0.1   nur die Person lernen (Masken), Hintergrund transparent
#   --test-cams (leer)                      alle Kameras trainieren, keine Testkamera
#   --traj-mode orbit                       Kreisfahrt um die Person als Video
# Weitere Argumente werden an train.py durchgereicht und überschreiben diese Werte.
#
# Aufruf: bash scripts/run_4danyone.sh <data_dir> <result_dir> [weitere Argumente für train.py]
set -euo pipefail

DATA=${1:?data_dir}
OUT=${2:?result_dir}
shift 2

bash "$(dirname "$0")/run_proto.sh" "$OUT" \
  --data-dir "$DATA" \
  --keyframe-step 1 \
  --lambda-reg 1e-2 --relocate --init-velocity knn --velocities-lr-final 5e-5 \
  --background random --lambda-mask 0.1 \
  --test-cams \
  --traj-mode orbit \
  "$@"
