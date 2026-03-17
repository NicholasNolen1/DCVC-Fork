# Training DCVC-RT Intra (Image) Model

This repo ships DCVC-RT compression/decompression code that loads pretrained weights, but does not include a training harness out of the box. The scripts in `Training/` add a minimal training workflow for the **intra (image / I-frame) codec** implemented by `src.models.image_model.DMCI`.

## What You Get

- `Training/train_dcvc_rt_intra.py`: trains `DMCI` on an image folder using a differentiable rate-distortion objective (STE quantization + likelihood-based rate estimates).
- Output checkpoints are **weights-only** and **compatible** with the existing project loader:
  - `src/utils/common.get_state_dict()` uses `torch.load(..., weights_only=True)` and accepts `{"state_dict": ...}`.

## Dataset Format

Provide a directory containing images (recursively), e.g.:

```
dataset_root/
  img001.png
  img002.jpg
  subdir/
    img003.png
```

Training uses random square crops (`--patch_size`) and converts RGB to BT.709 YCbCr internally to match the PNG evaluation path used by `test_video.py`.

Why crops:
- It is much cheaper than full images (bigger batch sizes, faster training).
- Learned codecs are typically trained on random patches for better generalization.

Why YCbCr:
- The repo’s PNG path converts RGB -> YCbCr before encoding.
- Training in the same space makes the trained weights behave more like inference.

Small images:
- Some COCO images are smaller than `--patch_size` (for example 240x320).
- The loader automatically upsizes images so a crop is always possible.

## Train

Example:

```bash
python Training/train_dcvc_rt_intra.py \
  --train_root /path/to/train_images \
  --val_root /path/to/val_images \
  --out_dir Training/runs/intra_mydata \
  --patch_size 256 \
  --batch_size 8 \
  --lr 1e-4 \
  --lambda_rd 0.01 \
  --max_steps 20000 \
  --device cuda
```

This produces checkpoints like:

`Training/runs/intra_mydata/dmci_intra_step0001000.pth`

## Use The Trained Weights With Existing Code

For any script that expects an intra model checkpoint path (for example `test_video.py --model_path_i ...`), point it at a generated `dmci_intra_step*.pth` file.

Notes:
- The weights-only checkpoints are intentionally minimal so `weights_only=True` loading works.
- The existing inference code calls `model.update(...)` after loading, which initializes entropy coding tables.

## Optional: Evaluate On An Image Folder With Real Bitstreams

If your environment has the codec extensions built (the same requirement as running normal encoding/decoding in this repo), you can measure bpp/PSNR on a folder:

```bash
python Training/eval_intra_folder.py \
  --weights Training/runs/intra_mydata/dmci_intra_step00020000.pth \
  --images /path/to/val_images \
  --qp 32 \
  --device cuda
```

Why a separate eval script:
- Training estimates bits using probability models (fast, differentiable).
- Real deployment uses the actual entropy coder and produces real bitstreams.
- This script tells you the real bpp/PSNR of your trained checkpoint.
