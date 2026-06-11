# Data Directory

This directory stores local datasets generated for development and screenshots.

Generated datasets are intentionally ignored by Git:

```text
data/simulated/
```

Smoke-test generation:

```bash
python scripts/generate_dataset.py --num 20 --out data/simulated/smoke --image-size 128 --seed 42
```

Each generated dataset contains:

- `images/*.png`: simulated diffraction images
- `metadata/*.json`: simulation parameters, artifact labels, metrics
- `labels.csv`: flat table for model training and report screenshots

