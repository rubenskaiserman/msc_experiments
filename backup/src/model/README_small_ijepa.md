# Small I-JEPA sanity-check implementation

Files:

- `small_ijepa.py`: drop-in model + LightningModule.
- `train_small_ijepa.py`: minimal CIFAR-10 training script using the existing `CIFAR10DataModule`.

This implementation is deliberately small:

- 32x32 CIFAR images.
- Patch size 4 by default, so the token grid is 8x8.
- Patch-token-only encoder, no `[CLS]` token during pretraining.
- Online context encoder + EMA target encoder.
- Narrow transformer predictor.
- Random non-overlapping square target blocks sampled inside the model.
- Labels are ignored.

Smoke test:

```bash
python small_ijepa.py
```

Example training:

```bash
python train_small_ijepa.py \
  --epochs 20 \
  --batch-size 256 \
  --accelerator auto \
  --devices auto
```

Useful sanity signals:

- `train_loss` should decrease from its initial value.
- `train_target_std` should not collapse toward zero.
- `train_pred_std` should become nonzero and track the target scale.
- If `val_loss` is unstable, reduce learning rate to `1e-4` and/or set `drop_path_rate=0` in `MiniIJEPAConfig`.

For a quick linear-probe-style check, use `MiniIJEPA.encode(images)` to obtain an average-pooled target-encoder representation.
