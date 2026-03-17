#!/usr/bin/env python3

"""
Evaluate a trained DMCI intra model on an image folder using the *real* codec path.

This script runs:
- RGB image -> BT.709 YCbCr
- DMCI.compress(...) to produce an actual bitstream (RANS entropy coding)
- DMCI.decompress(...) to reconstruct the image
- reports average bpp and PSNR over the folder

Because it uses the true entropy coder, it requires:
- CUDA (DMCI.compress uses torch.cuda events/streams)
- the codec extension modules built (same requirements as running `test_video.py` normally)

Why this script exists (even though training uses likelihoods):
- During training we estimate bits with probability models. That is fast and differentiable.
- At the end, you still want to know: "how big are the real bitstreams?"
- This script answers that by running the actual codec on real images.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def _repo_root() -> Path:
    # Resolve repository root as: <repo>/Training/eval_intra_folder.py -> parents[1] == <repo>
    return Path(__file__).resolve().parents[1]


def _add_repo_to_syspath() -> None:
    # Allow running this file directly without installing the package.
    import sys
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _list_images(root: Path, exts: Sequence[str]) -> List[Path]:
    # Recursively list images under `root` matching extensions.
    files: List[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in exts:
                files.append(p)
    files.sort()
    return files


def _pil_to_rgb_float(pil: Image.Image) -> torch.Tensor:
    # PIL -> CHW float32 in [0,1] RGB.
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    x = np.asarray(pil, dtype=np.uint8)  # HWC
    x_t = torch.from_numpy(x).to(dtype=torch.float32) / 255.0
    return x_t.permute(2, 0, 1).contiguous()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate DMCI intra codec on an image folder")
    p.add_argument("--weights", type=str, required=True, help="Weights-only checkpoint (.pth)")
    p.add_argument("--images", type=str, required=True, help="Image folder root")
    p.add_argument("--qp", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--extensions", type=str, default=".png,.jpg,.jpeg,.bmp,.webp")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    # Reduce noise if custom kernels are not built; evaluation itself still needs extensions.
    os.environ.setdefault("SUPPRESS_CUSTOM_KERNEL_WARNING", "1")
    _add_repo_to_syspath()

    # Local imports after sys.path is set so `python Training/eval_intra_folder.py` works.
    from src.models.image_model import DMCI  # pylint: disable=import-error
    from src.utils.common import get_state_dict  # pylint: disable=import-error
    from src.utils.transforms import rgb2ycbcr, ycbcr2rgb  # pylint: disable=import-error
    from src.utils.metrics import calc_psnr  # pylint: disable=import-error
    from src.layers.cuda_inference import replicate_pad  # pylint: disable=import-error

    args = _parse_args()
    qp = int(args.qp)

    device_str = str(args.device).lower()
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    if device.type != "cuda":
        raise SystemExit("This eval script requires CUDA (DMCI.compress uses torch.cuda events/streams).")

    # Load the model weights (compatible with repo helper which unwraps state_dict/net/module prefixes).
    net = DMCI()
    state = get_state_dict(args.weights)
    net.load_state_dict(state)
    net = net.to(device=device)
    net.eval()
    # Initializes entropy coding tables (requires codec extensions).
    net.update(force_zero_thres=None)

    exts = tuple(s.strip().lower() for s in str(args.extensions).split(",") if s.strip())
    files = _list_images(Path(args.images), exts)
    if int(args.limit) > 0:
        files = files[: int(args.limit)]
    if not files:
        raise SystemExit("No images found.")

    total_bpp = 0.0
    total_psnr = 0.0

    for p in tqdm(files, desc="eval", dynamic_ncols=True):
        with Image.open(p) as im:
            rgb = _pil_to_rgb_float(im)

        # Model expects BCHW float in [0,1]. Convert RGB to YCbCr to match repo evaluation path.
        x = rgb.unsqueeze(0).to(device=device)
        x_ycbcr = rgb2ycbcr(x)

        # DMCI operates on padded sizes aligned to 16 (latent stride).
        h, w = int(x_ycbcr.size(2)), int(x_ycbcr.size(3))
        padding_r, padding_b = DMCI.get_padding_size(h, w, 16)
        x_padded = replicate_pad(x_ycbcr, padding_b, padding_r)

        # Match the inference heuristic used in `test_video.py`:
        # use two entropy coders for >720p content to improve throughput.
        use_two_entropy_coders = (h * w) > (1280 * 720)
        net.set_use_two_entropy_coders(use_two_entropy_coders)

        # Real bitstream encode.
        enc = net.compress(x_padded, qp)
        bit_stream = enc["bit_stream"]

        # Real bitstream decode. The SPS fields are the ones DMCI.decompress reads.
        dec = net.decompress(
            bit_stream,
            sps={"height": h, "width": w, "ec_part": 1 if use_two_entropy_coders else 0, "use_ada_i": 0},
            qp=qp,
        )
        x_hat = dec["x_hat"][:, :, :h, :w]

        # Convert back to RGB for PSNR on the original images.
        rgb_hat = ycbcr2rgb(x_hat).clamp(0.0, 1.0)

        # Metrics in this repo use uint8 numpy arrays for PSNR.
        rgb_np = (rgb.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        rgb_hat_np = (rgb_hat.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        psnr = float(calc_psnr(rgb_np, rgb_hat_np))
        # bpp is computed on the original (unpadded) image size.
        bpp = (len(bit_stream) * 8.0) / float(h * w)

        total_psnr += psnr
        total_bpp += bpp

    n = float(len(files))
    print(f"images: {int(n)}")
    print(f"avg_bpp: {total_bpp / n:.6f}")
    print(f"avg_psnr: {total_psnr / n:.4f}")


if __name__ == "__main__":
    main()
