#!/bin/bash
# Stufe 0 der Diagnosen zur schnellen Bewegung (Ansaetze_schnelle_Bewegung.md, Abschnitt 5),
# Seilspringen mit 48 Ansichten, Test cam19 + cam40. Alle Läufe 30.000 Schritte.
#
#   D4  Seeds: Basislauf (run_4danyone.sh), ohne 4D-Regularisierung (noreg, Referenz für
#       D1/D3), Skelett-Prior 0,1 ohne Regularisierung (skelC), je Seeds 42/1/2.
#       Seed 42 von Basis und skelC liegen schon vor (48Views, 48Views_skel_l01_noreg).
#   D1  nur Frames 45–75 (--frame-start 45 --num-frames 31): jedes Bild wird ~21-mal statt
#       ~5-mal gesehen, die Relokation kann kein Budget in die ruhige Phase verschieben.
#       hullall: alle Oberflächenvoxel des Visual Hull als Start (~24.000 statt 8000 pro Frame).
#   D3  Pro-Frame-Modus: Dauer ≤ 0,4 Frames und μₜ fest (Beitrag zu Nachbarframes ≤ 4 %).
#
# Bereits fertige Läufe (ckpts/ckpt_final.pt vorhanden) werden übersprungen.
# Aufruf: bash scripts/run_stage0.sh [Lauf ...]   (ohne Argument: alle in der Reihenfolge unten)
set -uo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
DATA=~/masterarbeit/data/4danyone/48Views
HULLALL=~/masterarbeit/data/4danyone/48Views_hullall
OUT=~/masterarbeit/results/4danyone/st0

TEST="--test-cams cam19 cam40"
NOREG="--lambda-reg 0"
SKELC="--init-velocity skeleton --lambda-skel 0.1 --lambda-reg 0"
WIN="--frame-start 45 --num-frames 31 --no-render-traj"
PF="--max-duration-frames 0.4 --no-learn-times"

declare -A RUNS=(
  [noreg_s42]="$DATA|$NOREG --seed 42"
  [win_noreg]="$DATA|$NOREG $WIN"
  [win_hullall_noreg]="$HULLALL|$NOREG $WIN"
  [pf_noreg]="$DATA|$NOREG $PF"
  [win_pf_noreg]="$DATA|$NOREG $WIN $PF"
  [win_hullall_pf_noreg]="$HULLALL|$NOREG $WIN $PF"
  [noreg_s1]="$DATA|$NOREG --seed 1"
  [noreg_s2]="$DATA|$NOREG --seed 2"
  [skelC_s1]="$DATA|$SKELC --seed 1"
  [skelC_s2]="$DATA|$SKELC --seed 2"
  [base_s1]="$DATA|--seed 1"
  [base_s2]="$DATA|--seed 2"
)
ORDER=(noreg_s42 win_noreg win_hullall_noreg pf_noreg win_pf_noreg win_hullall_pf_noreg
       noreg_s1 noreg_s2 skelC_s1 skelC_s2 base_s1 base_s2)
[ $# -gt 0 ] && ORDER=("$@")

for name in "${ORDER[@]}"; do
  spec=${RUNS[$name]:?unbekannter Lauf $name}
  data=${spec%%|*}
  args=${spec#*|}
  if [ -f "$OUT/$name/ckpts/ckpt_final.pt" ]; then
    echo "== $name: schon fertig, übersprungen"
    continue
  fi
  echo "== $name ($(date +%H:%M)): $args"
  # shellcheck disable=SC2086
  bash "$REPO/scripts/run_4danyone.sh" "$data" "$OUT/$name" $TEST $args > /dev/null 2>&1 \
    || echo "!! $name fehlgeschlagen, siehe $OUT/$name/train.log"
  grep -h "^PSNR" "$OUT/$name/train.log" | tail -1
done
