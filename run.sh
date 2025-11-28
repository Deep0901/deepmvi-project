#!/usr/bin/env bash
set -euo pipefail
dataset=${1:-}
outdir=${2:-/tmp/output_deepmvi}

if [ -z "$dataset" ]; then
  echo "Usage: $0 <dataset> <output_dir>"
  exit 1
fi

mkdir -p "$outdir"

# run the main pipeline (adjust if your runner name differs)
python scripts/run_deepmvi_for_dataset.py --dataset "$dataset"

# export to ImputeGAP format
python scripts/export_to_imputegap.py "$dataset"

# copy outputs for the evaluator
cp -a output_imputegap/"$dataset"/* "$outdir"/

echo "Done. Outputs copied to $outdir"
