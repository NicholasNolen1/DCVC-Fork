#!/usr/bin/env python3

"""
Train the DCVC-RT intra (image / I-frame) codec weights.

This script trains `src.models.image_model.DMCI` using a standard learned-compression
rate-distortion objective:

  loss = (bpp_y + bpp_z) + lambda_rd * MSE(x, x_hat)

Key design choices (to match the existing repo behavior):
- Inputs are converted to BT.709 YCbCr during dataset loading (see `Training/data.py`),
  matching the PNG evaluation path in `test_video.py`.
- Checkpoints are saved as weights-only dicts: `{"state_dict": ...}` so they can be
  loaded with `src/utils/common.get_state_dict()` which uses `weights_only=True`.

This training loop does NOT require the codec entropy-coding C++/CUDA extensions,
because it uses likelihood-based rate estimates (see `Training/dmci_rd.py`) rather
than running RANS.

Plain-English summary:
- We show the model random image patches.
- The model produces a reconstruction `x_hat`.
- We estimate how many bits the latents would take (bpp_y + bpp_z).
- We also measure the reconstruction error (MSE).
- We optimize a weighted sum of those two things.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def _repo_root() -> Path:
    # Resolve repository root as: <repo>/Training/train_dcvc_rt_intra.py -> parents[1] == <repo>
    return Path(__file__).resolve().parents[1]


def _add_repo_to_syspath() -> None:
    # Allow running this file directly without installing the package.
    import sys
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train DCVC-RT intra (DMCI) weights")

    p.add_argument("--train_root", type=str, required=True, help="Image folder root")
    p.add_argument("--val_root", type=str, default=None, help="Optional validation image folder root")
    p.add_argument("--out_dir", type=str, default="Training/runs/intra", help="Output directory")

    # Data pipeline settings.
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)

    # Optimizer and objective settings.
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda_rd", type=float, default=0.01)
    p.add_argument("--max_steps", type=int, default=20000)

    # Rate control:
    # - if qp is fixed (>=0), train a single operating point
    # - if qp < 0, randomly sample qp per batch to cover the full range (0..63)
    p.add_argument("--qp", type=int, default=-1, help="Fixed qp; if <0, sample uniformly from [0,63]")

    # Device / reproducibility.
    p.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    p.add_argument("--seed", type=int, default=0)

    # Logging and checkpoint cadence.
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--val_every", type=int, default=1000)
    p.add_argument("--val_batches", type=int, default=25)

    return p.parse_args()


@torch.no_grad()
def _run_validation(
    model,
    val_loader: DataLoader,
    qp: int,
    device: torch.device,
    max_batches: int,
) -> dict:
    # Validation is run with torch.no_grad for speed and to avoid storing gradients.
    # Why: we only want numbers for logging, not parameter updates.
    model.eval()

    n = 0
    bpp_y = 0.0
    bpp_z = 0.0
    mse = 0.0
    loss = 0.0

    for batch in val_loader:
        # `batch` is CHW float in [0,1] (YCbCr by default) from `Training/data.py`.
        x = batch.to(device=device)
        out = model(x, qp=qp)
        bpp_y += float(out.bpp_y.item())
        bpp_z += float(out.bpp_z.item())
        mse += float(out.mse.item())
        loss += float(out.loss.item())
        n += 1
        if n >= max_batches:
            break

    if n == 0:
        return {"val_batches": 0}

    return {
        "val_batches": n,
        "val_bpp_y": bpp_y / n,
        "val_bpp_z": bpp_z / n,
        "val_bpp": (bpp_y + bpp_z) / n,
        "val_mse": mse / n,
        "val_loss": loss / n,
        "val_qp": int(qp),
    }


def _sample_qp(args: argparse.Namespace, device: torch.device) -> int:
    if int(args.qp) >= 0:
        return int(args.qp)
    # Uniform sampling across the model's supported QPs (0..63).
    # Why: it teaches one set of weights to work across the whole rate range.
    return int(torch.randint(0, 64, (1,), device=device).item())


def _save_weights_only(out_dir: Path, step: int, dmci) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"dmci_intra_step{step:07d}.pth"
    # Must be compatible with `torch.load(..., weights_only=True)` used by `src/utils/common.get_state_dict()`.
    #
    # Why weights-only:
    # - This repo's inference scripts want a plain state_dict (or state_dict nested under "state_dict"/"net").
    # - Keeping checkpoints minimal makes it easy to use them with the existing compressor/decompressor.
    # - If you want resumable training (optimizer state), you can extend this later.
    torch.save({"state_dict": dmci.state_dict()}, path)
    return path


def main() -> None:
    # Avoid printing the "cannot import cuda implementation ..." warning during training.
    os.environ.setdefault("SUPPRESS_CUSTOM_KERNEL_WARNING", "1")
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    _add_repo_to_syspath()

    # Local imports after sys.path is set so `python Training/train_dcvc_rt_intra.py` works.
    from Training.data import ImageFolderConfig, ImageFolderPatches  # pylint: disable=import-error
    from Training.dmci_rd import DMCITrainWrapper  # pylint: disable=import-error
    from src.models.image_model import DMCI  # pylint: disable=import-error

    args = _parse_args()

    # Seed torch RNG.
    # Why: makes runs more repeatable when you are debugging changes.
    torch.manual_seed(int(args.seed))

    # Device selection.
    # Why fallback: makes the script still runnable on machines without a GPU.
    device_str = str(args.device).lower()
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build training dataset (random crops).
    # Why shuffle: each step sees a different mixture of images/patches.
    train_cfg = ImageFolderConfig(
        root=Path(args.train_root),
        patch_size=int(args.patch_size),
        to_ycbcr=True,
        seed=int(args.seed),
    )
    train_ds = ImageFolderPatches(train_cfg)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        # Pinning helps host->GPU transfer when using CUDA.
        # Why: it can noticeably reduce input pipeline overhead.
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # Optional validation dataset.
    val_loader: Optional[DataLoader] = None
    if args.val_root:
        val_cfg = ImageFolderConfig(
            root=Path(args.val_root),
            patch_size=int(args.patch_size),
            to_ycbcr=True,
            seed=int(args.seed) + 1,
        )
        val_ds = ImageFolderPatches(val_cfg)
        val_loader = DataLoader(
            val_ds,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

    # Instantiate the base codec model and wrap it with a differentiable RD forward().
    #
    # Why the wrapper:
    # - DMCI's `compress()` is great for inference, but not differentiable.
    # - The wrapper computes a differentiable approximation for training.
    dmci = DMCI()
    dmci = dmci.to(device=device)
    model = DMCITrainWrapper(dmci, lambda_rd=float(args.lambda_rd)).to(device=device)

    # Basic optimizer.
    # Why Adam: common default for learned compression and stable for this kind of model.
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    # Write a small run config for reproducibility.
    # `default=str` handles Path objects in `train_cfg` and argparse Namespace.
    (out_dir / "run_config.json").write_text(
        __import__("json").dumps(asdict(train_cfg) | vars(args), indent=2, default=str),
        encoding="utf-8",
    )

    step = 0
    model.train()

    # We run "step-based" training rather than "epoch-based".
    # Why: the dataset is patch-sampled; "epochs" are less meaningful than a fixed step budget.
    pbar = tqdm(total=int(args.max_steps), desc="train", dynamic_ncols=True)
    loader_iter = iter(train_loader)

    while step < int(args.max_steps):
        try:
            batch = next(loader_iter)
        except StopIteration:
            # Restart the loader when it is exhausted.
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        # Transfer a batch to device.
        x = batch.to(device=device, non_blocking=True)
        # Either fixed or randomly sampled QP.
        qp = _sample_qp(args, device)

        # Forward computes x_hat, (bpp_y, bpp_z), mse, and loss.
        # Why these metrics: they are the pieces of the RD objective.
        out = model(x, qp=qp)

        # Standard PyTorch training step.
        optimizer.zero_grad(set_to_none=True)
        out.loss.backward()
        optimizer.step()

        step += 1
        pbar.update(1)

        if step % int(args.log_every) == 0:
            # tqdm postfix gives live training metrics on the console.
            pbar.set_postfix(
                qp=int(qp),
                loss=float(out.loss.item()),
                bpp=float((out.bpp_y + out.bpp_z).item()),
                mse=float(out.mse.item()),
            )

        if step % int(args.save_every) == 0:
            # Save a weights-only snapshot you can directly feed to `test_video.py --model_path_i`.
            _save_weights_only(out_dir, step, dmci)

        if val_loader is not None and step % int(args.val_every) == 0:
            # Validate at a fixed QP for comparable curves; default to mid-QP if sampling.
            val_qp = int(args.qp) if int(args.qp) >= 0 else 32
            stats = _run_validation(
                model=model,
                val_loader=val_loader,
                qp=val_qp,
                device=device,
                max_batches=int(args.val_batches),
            )
            # Append-only JSONL so it's easy to plot over time.
            with (out_dir / "val_log.jsonl").open("a", encoding="utf-8") as f:
                f.write(__import__("json").dumps({"step": step, **stats}) + "\n")
            # Return to training mode after validation.
            model.train()

    # Final checkpoint at the end of training.
    _save_weights_only(out_dir, step, dmci)
    pbar.close()


if __name__ == "__main__":
    main()
