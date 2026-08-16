"""Patch-based sliding-window inference for images larger than the training crop.

A U-Net trained on 128x128 patches cannot be handed a 2048x2048 slide: memory is
quadratic in the side length, and the network has never seen that field of view.
The standard answer -- and what nnU-Net does -- is to tile the image with
overlapping patches and blend the predictions.

The blending is the part that is easy to get wrong. Averaging patches uniformly
produces visible **seams**, because a pixel at the edge of a patch was predicted
with almost no surrounding context and is systematically less reliable than one
at the centre. Weighting each patch by a Gaussian centred on it makes the
contribution fall off toward the edges, so the confident centre of one patch
dominates the unreliable border of its neighbour.

The weights are accumulated alongside the predictions and divided out at the
end, which makes the scheme an exact partition of unity: every pixel's weights
sum to 1 regardless of how many patches covered it.
"""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def gaussian_weight_map(patch_size: Tuple[int, int], sigma_scale: float = 0.125,
                        device=None, dtype=torch.float32) -> torch.Tensor:
    """Gaussian window over a patch, peaking at the centre.

    ``sigma_scale`` is expressed as a fraction of the patch side, so the shape
    of the window is independent of patch size. The default of 1/8 is nnU-Net's.

    The minimum is clamped away from zero: a patch corner with weight exactly 0
    contributes nothing, and if that corner is the *only* coverage of some pixel
    -- which happens at the image border -- the accumulated weight there would be
    zero and the division would produce NaN.
    """
    height, width = patch_size
    coords_y = torch.arange(height, dtype=dtype, device=device) - (height - 1) / 2
    coords_x = torch.arange(width, dtype=dtype, device=device) - (width - 1) / 2
    sigma_y = max(height * sigma_scale, 1e-6)
    sigma_x = max(width * sigma_scale, 1e-6)

    gauss_y = torch.exp(-(coords_y ** 2) / (2 * sigma_y ** 2))
    gauss_x = torch.exp(-(coords_x ** 2) / (2 * sigma_x ** 2))
    window = gauss_y[:, None] * gauss_x[None, :]

    window = window / window.max()
    return window.clamp_min(1e-3)


def _patch_starts(length: int, patch: int, step: int) -> list[int]:
    """Start offsets covering ``length`` with the last patch flush to the end.

    Simply striding by ``step`` leaves an uncovered strip whenever the length is
    not a whole number of steps. Snapping the final patch to ``length - patch``
    guarantees full coverage; the extra overlap it creates is handled correctly
    by the weighted accumulation.
    """
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch + 1, step))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


@torch.no_grad()
def sliding_window_inference(
    image: torch.Tensor,
    model: Callable[[torch.Tensor], torch.Tensor],
    patch_size: Sequence[int] = (128, 128),
    overlap: float = 0.5,
    batch_size: int = 4,
    sigma_scale: float = 0.125,
    apply_sigmoid: bool = True,
) -> torch.Tensor:
    """Run ``model`` over overlapping patches and blend the result.

    Args:
        image: ``(B, C, H, W)``.
        model: anything mapping a patch batch to logits of the same spatial size.
        patch_size: the size the model was trained on.
        overlap: fraction of a patch shared with its neighbour. 0.5 is the usual
            choice; higher costs compute and buys smoother blending.
        apply_sigmoid: blend probabilities rather than logits. Logits from
            different patches are not on a comparable scale, so averaging them
            lets one overconfident patch dominate its neighbours.

    Returns:
        ``(B, n_classes, H, W)`` in the same spatial size as the input.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    batch, _, height, width = image.shape
    patch_h, patch_w = int(patch_size[0]), int(patch_size[1])
    device = image.device

    # An image smaller than the patch is padded up, then cropped back at the end.
    pad_h = max(0, patch_h - height)
    pad_w = max(0, patch_w - width)
    if pad_h or pad_w:
        # Reflection needs the pad to be strictly smaller than the dimension it
        # mirrors, which fails exactly when the image is much smaller than the
        # patch -- the case this branch exists for. Fall back to replication,
        # and then to zeros, rather than letting torch raise.
        if pad_h < height and pad_w < width:
            mode = "reflect"
        elif height > 1 and width > 1:
            mode = "replicate"
        else:
            mode = "constant"
        image = F.pad(image, [0, pad_w, 0, pad_h], mode=mode)
    padded_h, padded_w = image.shape[-2:]

    step_h = max(1, int(round(patch_h * (1 - overlap))))
    step_w = max(1, int(round(patch_w * (1 - overlap))))
    ys = _patch_starts(padded_h, patch_h, step_h)
    xs = _patch_starts(padded_w, patch_w, step_w)

    window = gaussian_weight_map((patch_h, patch_w), sigma_scale, device)

    accumulator: torch.Tensor | None = None
    weights = torch.zeros((1, 1, padded_h, padded_w), device=device,
                          dtype=torch.float32)

    coordinates = [(y, x) for y in ys for x in xs]
    for start in range(0, len(coordinates), batch_size):
        chunk = coordinates[start : start + batch_size]
        patches = torch.cat([
            image[:, :, y : y + patch_h, x : x + patch_w] for y, x in chunk
        ], dim=0)

        logits = model(patches)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]          # deep supervision returns a list
        predictions = torch.sigmoid(logits) if apply_sigmoid else logits

        if accumulator is None:
            accumulator = torch.zeros(
                (batch, predictions.shape[1], padded_h, padded_w),
                device=device, dtype=torch.float32,
            )

        for index, (y, x) in enumerate(chunk):
            piece = predictions[index * batch : (index + 1) * batch]
            accumulator[:, :, y : y + patch_h, x : x + patch_w] += piece * window
            weights[:, :, y : y + patch_h, x : x + patch_w] += window

    assert accumulator is not None
    output = accumulator / weights
    return output[:, :, :height, :width]


def estimate_patch_memory(patch_size: Sequence[int], base_channels: int,
                          depth: int, batch_size: int = 1) -> float:
    """Rough activation memory for one forward pass, in megabytes.

    Only the encoder/decoder feature maps are counted, which dominate. Useful
    for choosing a patch size before discovering the limit the hard way.
    """
    height, width = patch_size
    total_elements = 0
    for level in range(depth + 1):
        channels = base_channels * (2 ** level)
        level_h = height // (2 ** level)
        level_w = width // (2 ** level)
        # Two conv outputs per block, encoder and decoder.
        total_elements += 4 * channels * level_h * level_w
    return total_elements * batch_size * 4 / (1024 ** 2)
