# Training Guide

1. Organize datasets under `datasets/` with `real/` and `fake/` splits.
2. Generate a manifest:

```bash
python scripts/prepare_dataset.py
```

3. Train the model:

```bash
python scripts/train_model.py
```

The training pipeline includes:

- EfficientNetB0 transfer learning backbone
- Bidirectional LSTM temporal modeling
- Early stopping
- ReduceLROnPlateau
- Model checkpoints
- TensorBoard logging
- Mixed precision policy

Optional export tooling:

```bash
pip install -r backend/requirements-export.txt
```
