"""
Dataset helpers for training DCVC-RT Intra (DMCI) on a generic image folder.

The codec expects 3-channel inputs in [0, 1].

Why YCbCr:
- In this repo, the PNG/RGB testing path converts RGB -> BT.709 YCbCr before encoding.
- Training in the same color space helps ensure the learned weights behave the same at inference.

This module is intentionally lightweight:
- Only dependency is Pillow for image decoding.
- We return random *patches* (crops) to keep memory/IO manageable.

Why patches:
- Full-resolution images waste memory and reduce batch size.
- Random crops give a lot of variety and are the common way learned codecs are trained.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _list_images(root: Path, exts: Sequence[str]) -> List[Path]:
    """
    Collect image paths under `root`.

    Why this helper exists:
    - It gives us a stable, sorted file list (useful for reproducibility).
    - It keeps dataset logic separate from training logic.
    """
    files: List[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in exts:
                files.append(p)
        # If recursion is disabled, the caller should pass only the top directory.
    files.sort()
    return files


def _pil_to_chw_float(pil: Image.Image) -> torch.Tensor:
    """
    Convert a PIL image to a torch tensor:
    - shape: CHW
    - dtype: float32
    - value range: [0, 1]
    """
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    x = np.asarray(pil, dtype=np.uint8)  # HWC
    x_t = torch.from_numpy(x).to(dtype=torch.float32) / 255.0
    return x_t.permute(2, 0, 1).contiguous()


def _random_crop(x: torch.Tensor, patch_size: int, generator: Optional[torch.Generator]) -> torch.Tensor:
    """
    Uniform random crop of size (patch_size, patch_size) from a CHW tensor.
    """
    _, h, w = x.shape
    if h < patch_size or w < patch_size:
        raise ValueError(f"Image too small for patch_size={patch_size}: got {h}x{w}")

    if generator is None:
        top = int(torch.randint(0, h - patch_size + 1, (1,)).item())
        left = int(torch.randint(0, w - patch_size + 1, (1,)).item())
    else:
        top = int(torch.randint(0, h - patch_size + 1, (1,), generator=generator).item())
        left = int(torch.randint(0, w - patch_size + 1, (1,), generator=generator).item())

    return x[:, top:top + patch_size, left:left + patch_size]


@dataclass(frozen=True)
class ImageFolderConfig:
    """
    Configuration for `ImageFolderPatches`.

    Why patch_size must be divisible by 16:
    - DMCI downsamples by 16x in its analysis transform.
    - If the patch size is not aligned, the codec will pad internally, which changes
      the effective bitrate and can make training less consistent.
    """
    root: Path
    patch_size: int = 256
    recursive: bool = True
    extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    to_ycbcr: bool = True
    seed: Optional[int] = 0


class ImageFolderPatches(Dataset[torch.Tensor]):
    """
    Loads random patches from an image folder. Returns `CHW` float32 in [0,1].

    If `to_ycbcr=True`, converts BT.709 RGB -> YCbCr using `src.utils.transforms.rgb2ycbcr`,
    matching the codec's expected input path (see `test_video.py` PNG case).
    """

    def __init__(self, cfg: ImageFolderConfig):
        super().__init__()
        if cfg.patch_size % 16 != 0:
            raise ValueError("patch_size must be divisible by 16 (codec latent stride)")
        self.cfg = cfg
        self.root = Path(cfg.root)
        if not self.root.exists():
            raise FileNotFoundError(str(self.root))

        exts = tuple(e.lower() for e in cfg.extensions)
        # Respect `recursive` by limiting to top-level when disabled.
        #
        # Why keep this option:
        # - Sometimes your dataset folder contains many nested folders you do not want.
        # - For debugging, it is useful to point at a single flat directory.
        if cfg.recursive:
            self.files = _list_images(self.root, exts)
        else:
            self.files = []
            for name in os.listdir(self.root):
                p = self.root / name
                if p.is_file() and p.suffix.lower() in exts:
                    self.files.append(p)
            self.files.sort()
        if not self.files:
            raise ValueError(f"No images found under {self.root} with extensions {exts}")

        self._generator = None
        if cfg.seed is not None:
            # Deterministic crop positions across runs (useful when debugging).
            self._generator = torch.Generator()
            self._generator.manual_seed(int(cfg.seed))

        self._rgb2ycbcr = None
        if cfg.to_ycbcr:
            # Late import so this module can be imported before sys.path is set up by the CLI.
            from src.utils.transforms import rgb2ycbcr  # pylint: disable=import-error
            self._rgb2ycbcr = rgb2ycbcr

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Each item is one *random patch* from one image file.
        p = self.files[int(idx)]
        with Image.open(p) as im:
            x = _pil_to_chw_float(im)

        # Random crop is the main "augmentation" here.
        x = _random_crop(x, self.cfg.patch_size, self._generator)
        if self._rgb2ycbcr is not None:
            # Keep the training input in the same YCbCr space used by codec evaluation scripts.
            x = self._rgb2ycbcr(x)
        return x
