"""
Training-time rate-distortion forward pass for DCVC-RT intra (image) codec.

What this file is:
- The repo's `src.models.image_model.DMCI` is written for fast *inference*:
  it has `compress()` and `decompress()` that create a real bitstream using RANS
  entropy coding and hard rounding.

Why we need a separate training forward():
- `compress()` uses hard rounding and a real entropy coder. Those parts are not
  differentiable, so a standard PyTorch optimizer cannot learn from them directly.
- Training learned codecs is normally done with a differentiable *approximation*:
  we estimate the bitrate using probability models ("likelihoods") and we use a
  straight-through estimator (STE) for rounding.

What we do for training:
- Quantization: STE rounding (forward behaves like round, backward acts like identity).
- Rate estimate for z (hyper latents): `BitEstimator` CDF differences.
- Rate estimate for y (main latents): Gaussian CDF differences, following the same
  4-step masked (checkerboard) prior structure used in inference.

What you get out:
- The exact same model parameters are trained, and we save a normal PyTorch
  `state_dict`, so the weights plug back into the existing repo code.

Limits / expectations:
- The estimated bitrate from likelihoods will not exactly equal the true RANS
  bitstream size, but this is the standard approach used in learned compression.
- This forward() path does not need the C++/CUDA entropy coding extensions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn


def ste_round(x: torch.Tensor) -> torch.Tensor:
    """
    Straight-through estimator for rounding:
    - forward: round(x)
    - backward: identity (d/dx ~= 1)

    Why this is necessary:
    - Real rounding has zero/undefined gradient almost everywhere.
    - STE gives us a usable gradient so we can train with backprop.
    """
    return x + (torch.round(x) - x).detach()


def ste_round_and_clamp_to_int8_range(x: torch.Tensor) -> torch.Tensor:
    """
    Apply STE rounding and clamp to [-128, 127].

    Why the clamp is necessary:
    - The inference codec clamps symbols into the int8 range before entropy coding.
    - If training allows larger magnitudes, the learned distributions can drift into
      a range that inference will later clip, hurting quality and/or rate.
    """
    x_hat = ste_round(x)
    return torch.clamp(x_hat, -128.0, 127.0)


def gaussian_cdf(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Normal(0, scale) CDF evaluated at x.

    CDF(x) = 0.5 * (1 + erf(x / (scale * sqrt(2))))
    """
    return 0.5 * (1.0 + torch.erf(x / (scale * math.sqrt(2.0))))


def gaussian_likelihood_qres(qres: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Likelihood of a quantized residual under Normal(0, scale).

    If r is rounded to integer bins, then:
    P(r) = CDF(r + 0.5) - CDF(r - 0.5)
    """
    upper = gaussian_cdf(qres + 0.5, scale)
    lower = gaussian_cdf(qres - 0.5, scale)
    return upper - lower


def bpp_from_likelihood(likelihood: torch.Tensor, num_pixels: int) -> torch.Tensor:
    """
    Convert likelihoods to estimated bits-per-pixel for the *original* image size.

    bpp = sum_i -log2(p_i) / num_pixels
    """
    likelihood = torch.clamp(likelihood, min=1e-9)
    return (-torch.log2(likelihood).sum()) / float(num_pixels)


@dataclass(frozen=True)
class DMCIForwardOut:
    x_hat: torch.Tensor
    bpp_y: torch.Tensor
    bpp_z: torch.Tensor
    mse: torch.Tensor
    loss: torch.Tensor


class DMCITrainWrapper(nn.Module):
    """
    Wraps a `src.models.image_model.DMCI` instance with a differentiable forward()
    suitable for training.
    """

    def __init__(self, dmci: nn.Module, lambda_rd: float = 0.01):
        super().__init__()
        self.dmci = dmci
        self.lambda_rd = float(lambda_rd)

    def forward(self, x: torch.Tensor, qp: int) -> DMCIForwardOut:
        """
        Training forward pass.

        Inputs:
        - x: BCHW, float in [0, 1], 3 channels (YCbCr or RGB; repo uses YCbCr for codec paths)
        - qp: integer quantization parameter in [0, 63]

        Outputs:
        - reconstructed x_hat
        - estimated rate terms (bpp_y, bpp_z)
        - MSE distortion
        - total RD loss: (bpp_y + bpp_z) + lambda * MSE

        Why we take qp:
        - DCVC-RT supports many operating points with one model by conditioning on qp.
        - In this model qp selects learned scaling tensors (rate-control behavior).
        """
        if x.dim() != 4 or x.size(1) != 3:
            raise ValueError(f"Expected BCHW with 3 channels, got shape {tuple(x.shape)}")
        if not (0 <= int(qp) < self.dmci.get_qp_num()):
            raise ValueError(f"qp must be in [0, {self.dmci.get_qp_num()-1}], got {qp}")

        device = x.device
        dtype = x.dtype

        # Per-QP feature scaling (rate control).
        #
        # Why this exists:
        # - A single model supports a wide bitrate range.
        # - Different qp values pick different learned scales that trade off rate vs distortion.
        curr_q_enc = self.dmci.q_scale_enc[qp:qp + 1, :, :, :].to(device=device, dtype=dtype)
        curr_q_dec = self.dmci.q_scale_dec[qp:qp + 1, :, :, :].to(device=device, dtype=dtype)

        # Analysis transform: image -> latents y.
        # Why: learned transforms put the information into a space that can be entropy coded well.
        y = self.dmci.enc(x, curr_q_enc)
        y_pad = self.dmci.pad_for_y(y)

        # Hyperprior: y -> z (side information), then quantize z with STE.
        # Why: z helps predict the distribution of y, which improves compression.
        z = self.dmci.hyper_enc(y_pad)
        z_hat = ste_round_and_clamp_to_int8_range(z)

        # z rate (BitEstimator CDF differences).
        #
        # Why this is necessary:
        # - During training we need a differentiable "bits" estimate.
        # - We get P(z_hat) by taking CDF(z_hat+0.5) - CDF(z_hat-0.5), then bits = -log2(P).
        # - This is a standard way to turn a continuous CDF model into a discrete pmf estimate.
        index = torch.full((x.size(0),), int(qp), device=device, dtype=torch.long)
        z_likelihood = self.dmci.bit_estimator_z(z_hat + 0.5, index) - \
            self.dmci.bit_estimator_z(z_hat - 0.5, index)

        # Synthesize y distribution parameters from z_hat.
        #
        # Why:
        # - We cannot encode y efficiently without knowing (or predicting) its distribution.
        # - The hyper-decoder + fusion network predicts per-location parameters like scales/means.
        params = self.dmci.hyper_dec(z_hat)
        params = self.dmci.y_prior_fusion(params)
        _, _, y_h, y_w = y.shape
        params = params[:, :, :y_h, :y_w].contiguous()

        # Main latents (y) rate and quantization with the 4-step spatial prior.
        #
        # Why the 4-step masked structure:
        # - In inference, the codec entropy-codes y in 4 passes using checkerboard-style masks.
        # - Each pass can condition on the already-decoded parts without needing a slow autoregressive loop.
        # - We follow the same structure so training matches inference as closely as possible.
        y_hat, y_likelihood = self._rd_y_forward(y, params)

        # Synthesis transform: y_hat -> reconstructed image.
        # Why: map quantized latents back to pixel space.
        x_hat = self.dmci.dec(y_hat, curr_q_dec).clamp(0.0, 1.0)

        # Convert likelihoods into bpp terms normalized by the number of pixels in x.
        #
        # Why normalize:
        # - codec papers and plots typically report bitrate as bits-per-pixel.
        # - using bpp makes training comparable across patch sizes / resolutions.
        num_pixels = int(x.size(0) * x.size(2) * x.size(3))
        bpp_z = bpp_from_likelihood(z_likelihood, num_pixels)
        bpp_y = bpp_from_likelihood(y_likelihood, num_pixels)

        # Distortion term (mean squared error).
        # Why MSE: simple distortion proxy; can be swapped for MS-SSIM if desired.
        mse = torch.mean((x - x_hat) ** 2)

        # Total objective: rate + lambda * distortion.
        #
        # Why this form:
        # - It directly encodes the tradeoff we want: fewer bits vs better recon quality.
        loss = bpp_y + bpp_z + self.lambda_rd * mse

        return DMCIForwardOut(
            x_hat=x_hat,
            bpp_y=bpp_y,
            bpp_z=bpp_z,
            mse=mse,
            loss=loss,
        )

    def _rd_y_forward(self, y: torch.Tensor, common_params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable y quantization + likelihood estimation for the 4-pass spatial prior.

        In inference:
        - `compress_prior_4x` rounds residuals to int8 and entropy-codes them.
        In training (here):
        - we STE-round residuals
        - we estimate probability with a Gaussian model per symbol
        """
        q_enc, q_dec, scales, means = self.dmci.separate_prior(common_params, is_video=False)
        common_params = self.dmci.y_spatial_prior_reduction(common_params)

        # Keep scales within the same range used by the inference entropy coder scale table.
        #
        # Why clamp scales:
        # - Very small scales can make likelihoods numerically tiny (unstable gradients).
        # - Very large scales can make likelihoods too flat (poor rate estimates).
        scale_min = float(self.dmci.gaussian_encoder.scale_min)
        scale_max = float(self.dmci.gaussian_encoder.scale_max)

        dtype = y.dtype
        device = y.device
        b, c, h, w = y.size()
        mask_0, mask_1, mask_2, mask_3 = self.dmci.get_mask_4x(b, c, h, w, dtype, device)

        # Apply the quantization scaling (q_enc) before rounding residuals.
        # Why: this is how the inference path defines quantization for y.
        y_scaled = y * q_enc

        # Stage 0.
        y_hat_0, lik_0 = self._quant_and_likelihood(y_scaled, scales, means, mask_0, scale_min, scale_max)
        y_hat_so_far = y_hat_0

        # Stage 1.
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = self.dmci.y_spatial_prior(self.dmci.y_spatial_prior_adaptor_1(params)).chunk(2, 1)
        y_hat_1, lik_1 = self._quant_and_likelihood(y_scaled, scales, means, mask_1, scale_min, scale_max)
        y_hat_so_far = y_hat_so_far + y_hat_1

        # Stage 2.
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = self.dmci.y_spatial_prior(self.dmci.y_spatial_prior_adaptor_2(params)).chunk(2, 1)
        y_hat_2, lik_2 = self._quant_and_likelihood(y_scaled, scales, means, mask_2, scale_min, scale_max)
        y_hat_so_far = y_hat_so_far + y_hat_2

        # Stage 3.
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = self.dmci.y_spatial_prior(self.dmci.y_spatial_prior_adaptor_3(params)).chunk(2, 1)
        y_hat_3, lik_3 = self._quant_and_likelihood(y_scaled, scales, means, mask_3, scale_min, scale_max)
        # Apply decoder scaling q_dec after the full y_hat is reconstructed.
        # Why: inference multiplies by q_dec after decoding the masked stages.
        y_hat_scaled = (y_hat_so_far + y_hat_3) * q_dec

        # Combine per-stage likelihoods.
        #
        # Why multiply:
        # - Each stage "covers" a disjoint set of symbols (via masks), so the total likelihood
        #   is the product. Positions not coded in a stage use likelihood=1.0.
        y_likelihood = lik_0 * lik_1 * lik_2 * lik_3
        return y_hat_scaled, y_likelihood

    @staticmethod
    def _quant_and_likelihood(
        y_scaled: torch.Tensor,
        scales: torch.Tensor,
        means: torch.Tensor,
        mask: torch.Tensor,
        scale_min: float,
        scale_max: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For one masked stage:
        - compute masked residual (y - mean)
        - STE-round the residual (clamped to int8 range)
        - compute Gaussian likelihood under scale, and set lik=1.0 outside the mask
        - return the partial reconstruction (masked y_hat) and per-symbol likelihood
        """
        means_hat = means * mask
        scales_hat = scales * mask

        # Residual is only meaningful in masked positions; outside the mask it is forced to 0.
        #
        # Why:
        # - We only "encode" the masked locations in this stage.
        # - Non-masked locations are encoded in other stages.
        res = (y_scaled - means_hat) * mask
        qres = ste_round_and_clamp_to_int8_range(res) * mask

        # Likelihood for masked positions. For non-coded positions, use lik=1 so bits=0.
        #
        # Why abs() and clamp:
        # - The network outputs can be negative; scales must be positive.
        # - Clamping keeps the math stable and consistent with inference tables.
        scale = torch.clamp(scales_hat.abs(), min=scale_min, max=scale_max)
        lik = gaussian_likelihood_qres(qres, scale)
        lik = torch.clamp(lik, min=1e-9)
        lik = lik * mask + (1.0 - mask)

        # Partial reconstruction: only masked positions are populated for this stage.
        # The next stages condition on the sum of the previously reconstructed parts.
        y_hat = (qres + means_hat) * mask
        return y_hat, lik


def strip_to_inference_state_dict(ckpt: Dict) -> Dict[str, torch.Tensor]:
    """
    Accept either:
    - a raw `state_dict`
    - a checkpoint dict with `state_dict` or `net`
    Returns a plain state_dict suitable for `DMCI.load_state_dict()`.
    """
    if not isinstance(ckpt, dict):
        raise TypeError("checkpoint must be a dict")

    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    if "net" in ckpt and isinstance(ckpt["net"], dict):
        return ckpt["net"]
    # Assume already a state_dict.
    return ckpt
