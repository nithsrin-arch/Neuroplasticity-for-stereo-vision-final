# Self Supervised Stereo Refinement

This module trains and evaluates a self-supervised stereo disparity refinement model on KITTI-style data.
It wraps a base stereo network (PSMNet) and learns a correction head that improves disparity robustness under degradations.

## Project Structure

- `train.py` - main training script
- `train_enhanced.py` - stricter training script with required paths
- `test.py` - evaluation on clean vs degraded validation split, plus metrics/plots
- `infer.py` - single stereo-pair inference
- `configs/default.py` - default hyperparameters and data/degradation settings

## Requirements

- Python 3.9+
- PyTorch
- NumPy
- Pillow
- Matplotlib (used by `test.py`)
- A local checkout of PSMNet (path passed via `--psmnet-dir`)

Install the common Python dependencies (example):

```bash
pip install torch torchvision numpy pillow matplotlib
```

## Dataset Layout

The code expects KITTI training-style folders under a `data_root` that contains:

- `image_2/` (left RGB)
- `image_3/` (right RGB)
- `disp_occ_0/` (ground-truth disparity)

`test.py` and training utilities auto-detect the KITTI training folder beneath `--data-root`.

## Training

### Option 1: Standard training

```bash
python train.py --data-root "D:/path/to/KITTI" --psmnet-dir "D:/path/to/PSMNet" --out-dir outputs
```

Common flags:

- `--epochs` (default: `50`)
- `--batch-size` (default: `1`)
- `--lr` (default: `1e-4`)
- `--debug` (prints per-batch diagnostics)
- `--show` (display disparity visualizations while running)

### Option 2: Enhanced training

```bash
python train_enhanced.py --data-root "D:/path/to/KITTI" --psmnet-dir "D:/path/to/PSMNet" --out-dir outputs_enhanced
```

This variant requires explicit path arguments and is useful for cleaner experiment setup.

## Evaluation

Run evaluation using a trained checkpoint:

```bash
python test.py --data-root "D:/path/to/KITTI" --psmnet-dir "D:/path/to/PSMNet" --checkpoint "outputs/best.pt" --output-dir outputs/test_disparity
```

Optional degradation controls:

- `--degrade-type` = `blur` | `noise` | `occlusion`
- `--degrade-severity` (integer, default `4`)
- `--degrade-camera` = `left` | `right`

Outputs include:

- predicted disparity maps
- JSON metrics (EPE, D1-all)
- clean vs degraded comparison plots

## Inference (Single Pair)

```bash
python infer.py --left "sample/left.png" --right "sample/right.png" --checkpoint "outputs/best.pt" --out outputs/infer_disparity --psmnet-dir "D:/path/to/PSMNet"
```

To also run degraded inference:

```bash
python infer.py --left "sample/left.png" --right "sample/right.png" --checkpoint "outputs/best.pt" --out outputs/infer_disparity --degrade --degrade-type noise --degrade-severity 4 --psmnet-dir "D:/path/to/PSMNet"
```

## Notes

- If you see path errors, verify `--data-root` and `--psmnet-dir` first.
- Default crop size is `256x512` (`configs/default.py`).
- GPU is used automatically when CUDA is available; otherwise CPU is used.
