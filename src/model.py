"""U-Net for 2D medical image segmentation.

The architecture that still wins most medical segmentation benchmarks, and the
reason is the skip connections. An encoder-decoder without them has to
reconstruct precise boundaries from a heavily downsampled bottleneck, which it
cannot do -- outputs come out blobby and boundaries land several pixels off.
The skips carry full-resolution spatial detail straight across, so the decoder
only has to decide *what* is where, not re-derive *where*.

Medical-specific choices baked in:

* **InstanceNorm, not BatchNorm.** Batch sizes are tiny (often 2-4) because the
  images are large, and BatchNorm's statistics are unusable at that size.
  InstanceNorm normalises per-sample and is batch-size independent.
* **Bilinear upsampling by default** rather than transposed convolution, which
  produces the checkerboard artefacts that look like texture and are not.
* **Deep supervision** optionally attaches heads to intermediate decoder levels,
  which speeds convergence on small datasets by giving gradient a short path.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with normalisation, activation and dropout."""

    def __init__(self, in_channels: int, out_channels: int,
                 dropout: float = 0.0, norm: str = "instance") -> None:
        super().__init__()

        def make_norm(channels: int) -> nn.Module:
            if norm == "instance":
                return nn.InstanceNorm2d(channels, affine=True)
            if norm == "batch":
                return nn.BatchNorm2d(channels)
            if norm == "group":
                return nn.GroupNorm(min(8, channels), channels)
            if norm == "none":
                return nn.Identity()
            raise ValueError(f"unknown norm '{norm}'")

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            make_norm(out_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            make_norm(out_channels),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    """Additive attention gate on a skip connection (Oktay et al., 2018).

    The plain U-Net concatenates the entire encoder feature map into the
    decoder, background included. When the foreground is under 1% of the image
    that is overwhelmingly irrelevant signal, and the decoder has to learn to
    ignore it.

    The gate instead computes a spatial attention map from the *coarser*
    decoder features -- which already know roughly where the object is -- and
    multiplies the skip by it before concatenation. Regions the decoder has no
    interest in are suppressed at source.

    ``g`` is the gating signal from the decoder, ``x`` the skip connection.
    """

    def __init__(self, gate_channels: int, skip_channels: int,
                 inter_channels: int | None = None) -> None:
        super().__init__()
        inter_channels = inter_channels or max(1, skip_channels // 2)
        self.theta = nn.Conv2d(skip_channels, inter_channels, 1, bias=False)
        self.phi = nn.Conv2d(gate_channels, inter_channels, 1, bias=True)
        self.psi = nn.Conv2d(inter_channels, 1, 1, bias=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the gated skip and the attention map, for inspection."""
        # The gating signal may be coarser than the skip; bring it up first.
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear",
                                 align_corners=False)
        attention = torch.sigmoid(
            self.psi(F.relu(self.theta(skip) + self.phi(gate)))
        )
        return skip * attention, attention


class UNet(nn.Module):
    """U-Net with a configurable depth and channel schedule."""

    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 1,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.0,
        norm: str = "instance",
        bilinear: bool = True,
        deep_supervision: bool = False,
        attention: bool = False,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.n_classes = n_classes
        self.deep_supervision = deep_supervision
        self.attention = attention

        channels = [base_channels * (2 ** i) for i in range(depth + 1)]

        self.encoders = nn.ModuleList()
        previous = in_channels
        for level in range(depth):
            self.encoders.append(ConvBlock(previous, channels[level], dropout, norm))
            previous = channels[level]
        self.bottleneck = ConvBlock(previous, channels[depth], dropout, norm)

        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.gates = nn.ModuleList() if attention else None
        for level in reversed(range(depth)):
            if attention:
                # Gating signal is the upsampled decoder feature, which has
                # already been projected to channels[level].
                self.gates.append(
                    AttentionGate(channels[level], channels[level])
                )
            if bilinear:
                # Upsample + 1x1 conv: no checkerboard artefacts, fewer params.
                self.upsamples.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear",
                                align_corners=False),
                    nn.Conv2d(channels[level + 1], channels[level], 1),
                ))
            else:
                self.upsamples.append(nn.ConvTranspose2d(
                    channels[level + 1], channels[level], 2, stride=2
                ))
            # Input is the upsampled features concatenated with the skip.
            self.decoders.append(
                ConvBlock(channels[level] * 2, channels[level], dropout, norm)
            )

        self.head = nn.Conv2d(base_channels, n_classes, 1)
        if deep_supervision:
            self.auxiliary_heads = nn.ModuleList([
                nn.Conv2d(channels[level], n_classes, 1)
                for level in reversed(range(1, depth))
            ])

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """Logits at full resolution, or a list of them under deep supervision."""
        skips: List[torch.Tensor] = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)

        auxiliary: List[torch.Tensor] = []
        attention_maps: List[torch.Tensor] = []
        for index, (upsample, decoder) in enumerate(
            zip(self.upsamples, self.decoders)
        ):
            x = upsample(x)
            skip = skips[-(index + 1)]
            # Odd input sizes make the upsampled tensor a pixel smaller than the
            # skip; pad rather than crop so no border information is discarded.
            if x.shape[-2:] != skip.shape[-2:]:
                diff_y = skip.shape[-2] - x.shape[-2]
                diff_x = skip.shape[-1] - x.shape[-1]
                x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                              diff_y // 2, diff_y - diff_y // 2])

            if self.gates is not None:
                skip, attention = self.gates[index](x, skip)
                attention_maps.append(attention)

            x = decoder(torch.cat([skip, x], dim=1))
            if (self.deep_supervision and self.training
                    and index < len(self.upsamples) - 1):
                auxiliary.append(self.auxiliary_heads[index](x))

        logits = self.head(x)
        if return_attention:
            return logits, attention_maps
        if self.deep_supervision and self.training and auxiliary:
            return [logits] + auxiliary
        return logits

    @torch.no_grad()
    def predict(self, x: torch.Tensor, tta: bool = False,
                threshold: float = 0.5) -> torch.Tensor:
        """Binary mask prediction, optionally with flip TTA."""
        probabilities = self.predict_proba(x, tta)
        return (probabilities >= threshold).float()

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor, tta: bool = False) -> torch.Tensor:
        """Foreground probability map.

        TTA averages over the dihedral flips. Unlike natural images, flipping a
        medical scan is label-preserving for most modalities, so all four views
        are legitimate -- but the inverse flip must be applied to each output
        before averaging, or the views cancel each other out into mush.
        """
        was_training = self.training
        self.eval()
        try:
            logits = self(x)
            if isinstance(logits, list):
                logits = logits[0]
            probabilities = torch.sigmoid(logits)

            if tta:
                accumulated = [probabilities]
                for dims in ([-1], [-2], [-1, -2]):
                    flipped = self(torch.flip(x, dims=dims))
                    if isinstance(flipped, list):
                        flipped = flipped[0]
                    accumulated.append(
                        torch.flip(torch.sigmoid(flipped), dims=dims)
                    )
                probabilities = torch.stack(accumulated).mean(0)
            return probabilities
        finally:
            self.train(was_training)


def build_model(cfg: dict) -> UNet:
    m = cfg["model"]
    return UNet(
        in_channels=m["in_channels"], n_classes=m["n_classes"],
        base_channels=m["base_channels"], depth=m["depth"],
        dropout=m["dropout"], norm=m["norm"], bilinear=m["bilinear"],
        deep_supervision=m["deep_supervision"],
        attention=m.get("attention", False),
    )
