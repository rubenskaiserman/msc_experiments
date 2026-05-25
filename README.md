# msc_experiments

`video_mae_thewell.ipynb` has been split into a Python project under `src/thewell_videomae/` with CLI scripts in `scripts/`.

Main entrypoints:

- `python scripts/download_thewell.py --data-base /path/to/the_well_data`
- `python scripts/train_videomae.py --data-base /path/to/the_well_data --output-dir runs/videomae_shear_flow`
- `python scripts/train_probe.py --pretrained-checkpoint runs/videomae_shear_flow/checkpoints/<ckpt>.ckpt --data-base /path/to/the_well_data --output-dir runs/videomae_probe`
- `python scripts/evaluate_probe.py --probe-checkpoint runs/videomae_probe/checkpoints/<ckpt>.ckpt --output-dir runs/videomae_probe_eval`

Project layout:

- `src/thewell_videomae/config.py`: experiment and trainer configuration
- `src/thewell_videomae/data.py`: The Well download logic, dataset wrappers, normalization, LightningDataModule
- `src/thewell_videomae/model.py`: VideoMAE backbone and attentive probe
- `src/thewell_videomae/modules.py`: Lightning modules for pretraining and frozen-probe training
- `src/thewell_videomae/metrics.py`: reconstruction and downstream metrics
- `scripts/*.py`: runnable CLIs
