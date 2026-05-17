# CIFAR-10 ViT Lightning Baseline

Small refactor of the single-file Lightning CIFAR-10 ViT script. The goal is to keep the project readable while making it easier to replace attention blocks, add JEPA pretraining, or insert randomized linear algebra tooling later.

## Structure

```text
vit_cifar10_lightning_refactor/
├── train.py                       # CLI, logger, callbacks, Trainer setup
├── requirements.txt
└── vit_cifar10/
    ├── __init__.py
    ├── data.py                    # CIFAR-10 DataModule and transforms
    ├── lightning_compat.py        # lightning.pytorch / pytorch_lightning fallback imports
    ├── lit_module.py              # Lightning training/val/test logic
    ├── model.py                   # ViT architecture only
    └── utils.py                   # optimizer groups, scheduler, metrics, parsing helpers
```

## Install

```bash
pip install -r requirements.txt
```

If TensorBoard complains about `pkg_resources`, use:

```bash
python -m pip install "setuptools<81"
```

## Train

```bash
python train.py --epochs 50 --batch-size 128 --amp --randaugment --random-erasing 0.25
```

If your RTX 4050 runs out of memory:

```bash
python train.py --epochs 50 --batch-size 64 --amp --randaugment --random-erasing 0.25
```

Smoke test:

```bash
python train.py --fast-dev-run
```

## TensorBoard

```bash
tensorboard --logdir ./runs/lightning_vit_cifar10 --port 6006
```

Then open <http://localhost:6006>.
