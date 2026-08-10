"""Synthetic medical-style images with lesion masks.

Built to reproduce the properties that make medical segmentation hard, rather
than clean shapes on a black background:

* **Severe class imbalance** -- lesions occupy roughly 0.3-3% of the image.
* **Low contrast** -- lesion intensity overlaps the surrounding tissue, so the
  boundary is genuinely ambiguous, as it is on a real scan.
* **Correlated texture noise** -- speckle and intensity inhomogeneity, not
  independent Gaussian pixels which a 3x3 filter removes trivially.
* **Empty cases** -- a fraction of images contain no lesion at all. Real
  screening cohorts are mostly negative, and a model never shown a negative
  learns to always find something.
* **Bias field** -- the smooth multiplicative intensity drift characteristic of
  MRI, which defeats any fixed global threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SegmentationSample:
    image: np.ndarray            # (H, W) float32
    mask: np.ndarray             # (H, W) float32, 0 or 1
    has_lesion: bool


def _smooth_noise(shape: Tuple[int, int], scale: float,
                  rng: np.random.Generator) -> np.ndarray:
    """Low-frequency field, made by upsampling coarse noise bilinearly."""
    small = max(2, int(min(shape) / max(scale, 1e-6)))
    coarse = rng.normal(size=(small, small))
    tensor = torch.from_numpy(coarse).float()[None, None]
    upsampled = torch.nn.functional.interpolate(
        tensor, size=shape, mode="bilinear", align_corners=False
    )
    return upsampled[0, 0].numpy()


def make_sample(
    size: int = 128,
    lesion_probability: float = 0.75,
    min_lesion_fraction: float = 0.003,
    max_lesion_fraction: float = 0.03,
    contrast: float = 0.35,
    noise_level: float = 0.12,
    seed: int = 0,
) -> SegmentationSample:
    """One synthetic scan slice with its ground-truth mask."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)

    # Background anatomy: a bright elliptical "organ" on darker surroundings.
    centre_y, centre_x = size / 2 + rng.normal(0, size * 0.04, 2)
    radius_y = size * rng.uniform(0.30, 0.40)
    radius_x = size * rng.uniform(0.30, 0.40)
    organ = (((yy - centre_y) / radius_y) ** 2
             + ((xx - centre_x) / radius_x) ** 2) <= 1.0

    image = np.full((size, size), 0.15, dtype=np.float32)
    image[organ] = 0.55
    # Internal tissue texture, correlated rather than per-pixel independent.
    image += 0.08 * _smooth_noise((size, size), size / 12, rng)

    mask = np.zeros((size, size), dtype=np.float32)
    has_lesion = rng.random() < lesion_probability

    if has_lesion:
        target_fraction = rng.uniform(min_lesion_fraction, max_lesion_fraction)
        target_pixels = target_fraction * size * size
        radius = float(np.sqrt(target_pixels / np.pi))

        # Place the lesion inside the organ, so it cannot be found by position.
        organ_pixels = np.argwhere(organ)
        for _ in range(50):
            ly, lx = organ_pixels[rng.integers(len(organ_pixels))]
            if (((ly - centre_y) / radius_y) ** 2
                    + ((lx - centre_x) / radius_x) ** 2) < 0.55:
                break

        # Irregular boundary via a radius that varies with angle -- a perfect
        # circle would be findable by template matching.
        angles = np.arctan2(yy - ly, xx - lx)
        distances = np.sqrt((yy - ly) ** 2 + (xx - lx) ** 2)
        wobble = 1.0 + 0.25 * np.sin(3 * angles + rng.uniform(0, 6.28)) \
            + 0.15 * np.sin(5 * angles + rng.uniform(0, 6.28))
        mask = (distances <= radius * wobble).astype(np.float32)
        # Lesion contrast is modest and varies, so some cases are genuinely hard.
        image += mask * contrast * rng.uniform(0.6, 1.4)

    # Bias field: smooth multiplicative drift, the MRI artefact that breaks
    # any fixed global intensity threshold.
    image *= 1.0 + 0.25 * _smooth_noise((size, size), size / 3, rng)
    # Speckle.
    image += noise_level * rng.normal(size=(size, size))

    return SegmentationSample(
        image=np.clip(image, 0, 1).astype(np.float32),
        mask=mask,
        has_lesion=bool(has_lesion and mask.sum() > 0),
    )


class SegmentationDataset(Dataset):
    """Generates samples on the fly, with optional augmentation."""

    def __init__(
        self,
        n_samples: int,
        size: int = 128,
        seed: int = 0,
        augment: bool = False,
        lesion_probability: float = 0.75,
        **sample_kwargs,
    ) -> None:
        self.n_samples = n_samples
        self.size = size
        self.seed = seed
        self.augment = augment
        self.lesion_probability = lesion_probability
        self.sample_kwargs = sample_kwargs

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = make_sample(
            self.size, self.lesion_probability,
            seed=self.seed * 1_000_003 + idx, **self.sample_kwargs,
        )
        image, mask = sample.image, sample.mask

        if self.augment:
            rng = np.random.default_rng(self.seed * 7919 + idx)
            # Flips and 90-degree rotations only: they are exactly
            # label-preserving and need no interpolation, so the mask stays
            # binary. Rotating by an arbitrary angle would blur mask edges into
            # fractional values.
            if rng.random() < 0.5:
                image, mask = image[:, ::-1], mask[:, ::-1]
            if rng.random() < 0.5:
                image, mask = image[::-1], mask[::-1]
            k = int(rng.integers(4))
            if k:
                image, mask = np.rot90(image, k), np.rot90(mask, k)
            # Intensity jitter applies to the image only, never the mask.
            image = np.clip(image * rng.uniform(0.9, 1.1)
                            + rng.uniform(-0.05, 0.05), 0, 1)

        return (
            torch.from_numpy(np.ascontiguousarray(image)).float().unsqueeze(0),
            torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0),
        )


def dataset_statistics(dataset: SegmentationDataset, n: int = 200) -> dict:
    """Foreground fraction and empty-case rate, for sanity-checking a run."""
    fractions, empty = [], 0
    for i in range(min(n, len(dataset))):
        _, mask = dataset[i]
        fraction = float(mask.mean())
        fractions.append(fraction)
        empty += fraction == 0.0
    return {
        "mean_foreground_fraction": float(np.mean(fractions)),
        "max_foreground_fraction": float(np.max(fractions)),
        "empty_fraction": empty / max(1, min(n, len(dataset))),
    }


def build_datasets(cfg: dict):
    d = cfg["data"]
    common = dict(
        size=d["size"], lesion_probability=d["lesion_probability"],
        min_lesion_fraction=d["min_lesion_fraction"],
        max_lesion_fraction=d["max_lesion_fraction"],
        contrast=d["contrast"], noise_level=d["noise_level"],
    )
    return (
        SegmentationDataset(d["n_train"], seed=cfg["seed"],
                            augment=d["augment"], **common),
        SegmentationDataset(d["n_val"], seed=cfg["seed"] + 50_000,
                            augment=False, **common),
        SegmentationDataset(d["n_test"], seed=cfg["seed"] + 90_000,
                            augment=False, **common),
    )
