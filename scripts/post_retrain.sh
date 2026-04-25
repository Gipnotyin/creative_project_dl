#!/usr/bin/env bash
# Run after train.py finishes a clean retrain on configs/baseline.yaml.
# Refreshes all evaluation/inference artifacts so they match the new best.pt.

set -euo pipefail

cd "$(dirname "$0")/.."

echo "[1/7] evaluate_detailed val"
python -m src.similis_baseline.evaluate_detailed \
  --checkpoint artifacts/checkpoints/best.pt \
  --split val \
  --output-dir artifacts/reports/val_detailed

echo "[2/7] evaluate_detailed test"
python -m src.similis_baseline.evaluate_detailed \
  --checkpoint artifacts/checkpoints/best.pt \
  --split test \
  --output-dir artifacts/reports/test_detailed

echo "[3/7] predict on raw images"
python -m src.similis_baseline.predict \
  --checkpoint artifacts/checkpoints/best.pt \
  --input-dir data/raw/images \
  --output artifacts/preds/inference.csv

echo "[4/7] error_factor_report val"
python -m src.similis_baseline.error_factor_report \
  --pred-csv artifacts/reports/val_detailed/predictions.csv \
  --split-name val \
  --output-dir artifacts/reports/val_error_factors

echo "[5/7] error_factor_report test"
python -m src.similis_baseline.error_factor_report \
  --pred-csv artifacts/reports/test_detailed/predictions.csv \
  --split-name test \
  --output-dir artifacts/reports/test_error_factors

echo "[6/7] cautious_examples val + test"
python -m src.similis_baseline.cautious_examples \
  --checkpoint artifacts/checkpoints/best.pt \
  --pred-csv artifacts/reports/val_detailed/predictions.csv \
  --output-dir artifacts/reports/val_cautious_examples
python -m src.similis_baseline.cautious_examples \
  --checkpoint artifacts/checkpoints/best.pt \
  --pred-csv artifacts/reports/test_detailed/predictions.csv \
  --output-dir artifacts/reports/test_cautious_examples

echo "[7/7] train_log_report"
python -m src.similis_baseline.train_log_report \
  --train-log artifacts/reports/train_log.csv \
  --output-dir artifacts/reports/train_log_report

echo "Done. Refresh README/REPORT next."
