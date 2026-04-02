python -m src.similis_baseline.predict \
  --checkpoint artifacts/checkpoints/best.pt \
  --input-dir data/raw/images \
  --output artifacts/preds/inference.csv
