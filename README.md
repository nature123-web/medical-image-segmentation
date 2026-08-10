# Medical Image Segmentation with U-Net

Lesion segmentation with a U-Net built from scratch, focused on the thing that
actually decides whether a medical segmentation model works: **extreme class
imbalance**, and the losses and metrics that survive it.

A lesion often occupies under 1% of a scan. Plain cross-entropy is minimised by
predicting background everywhere — the loss falls, pixel accuracy reads 99%+,
and the Dice score is zero. Almost every design decision here follows from that.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.train --config configs/base.yaml
python -m src.train --config configs/base.yaml --loss bce_only   # watch it fail
python -m src.predict --checkpoint runs/base/best.pt --plot overlay.png
```

The run prints this before training starts, so the baseline is never in doubt:

```
mean foreground: 1.284%  empty cases: 24.5%
note: predicting all-background would score 98.72% pixel accuracy
      and 0 Dice on lesion cases
```

## Losses

| Loss | What it fixes |
| --- | --- |
| **Dice** | region overlap; directly optimises the reported metric |
| **BCE** | per-pixel calibration; stabilises Dice's gradient when the prediction is nearly empty |
| **Focal** | down-weights the vast majority of easy background pixels |
| **Tversky** | independent false-positive/false-negative weights — raise `beta` to buy recall |

The default is Dice + BCE, and that combination is standard because each covers
the other's failure mode: Dice alone has unstable gradients on a near-empty
prediction (the denominator collapses) and says nothing about calibration; BCE
alone ignores region structure entirely.

**Dice is computed per sample, not pooled over the batch.** This matters more
than it sounds, and there is a test for it: pooling lets one large structure
dominate the denominator so a completely missed small lesion barely registers.

```
pooled over batch:  0.94   ← the missed lesion is invisible
per sample:         0.51   ← the truth
```

`TverskyLoss(alpha=0.3, beta=0.7)` is available for screening applications where
a missed lesion costs far more than a false positive a radiologist dismisses in
two seconds.

## Metrics

Overlap metrics and boundary metrics fail differently, so both are reported:

- **Dice / IoU** — region overlap.
- **HD95** — 95th-percentile symmetric surface distance. A prediction can have
  excellent Dice and a terrible boundary; one spurious region far from the
  lesion barely moves Dice and is exactly what a clinician notices. The 95th
  percentile rather than the maximum stops a single stray pixel defining the
  metric — a behaviour with its own test.
- **Case-level sensitivity / precision** — did it find the lesion *at all*.
  This is how screening tools are judged in practice, before boundary quality
  is even discussed.

**Dice is reported twice**: over all cases and over lesion-bearing cases only.
They differ substantially when the dataset contains negatives, because an
empty-and-correct prediction scores 1.0 and inflates the overall mean. Both
numbers are needed — the first describes behaviour in a screening cohort, the
second describes segmentation quality.

## Architecture choices

- **InstanceNorm, not BatchNorm.** Medical images are large so batches are tiny
  (2–8). BatchNorm statistics are unusable at that size. There is a test that
  the model trains with batch size 1.
- **Bilinear upsampling** by default instead of transposed convolution, which
  produces checkerboard artefacts that look like texture and are not.
- **Padding, not cropping**, when a skip connection and the upsampled tensor
  disagree by a pixel on odd input sizes — cropping silently discards border
  information. Tested at 50×50.
- **Deep supervision** (optional) attaches heads to intermediate decoder levels,
  which speeds convergence on small datasets.

## Post-processing and thresholding

**Threshold tuning.** 0.5 is only optimal for balanced classes. At ~1% foreground
the model is systematically under-confident and the best operating point is well
below 0.5. The run tunes it on validation and reports both.

**Small-component removal.** A model whose output is scattered with 3-pixel
specks can have good Dice and be unusable — every speck is an alert someone must
dismiss. Connected-component filtering is applied and the effect reported
separately, so you can see what it bought.

**TTA** averages the four dihedral flips. Unlike natural images these are
label-preserving for most modalities. The inverse flip is applied to each output
before averaging; forget that and the views cancel into mush, which a test pins
using a hand-built flip-equivariant model.

## The synthetic data is built to be hard

A generator making bright circles on black would be solved by a threshold. This
one produces:

- **0.3–3% foreground** — genuine imbalance
- **low, variable contrast** — the lesion boundary is honestly ambiguous
- **correlated texture noise**, not per-pixel Gaussian a 3×3 filter removes
- **~25% empty cases** — a model never shown a negative learns to always find
  something
- **a bias field** — the smooth multiplicative intensity drift characteristic of
  MRI, which defeats any fixed global threshold

That last point is asserted directly: a test sweeps every global threshold and
requires the best achievable Dice to stay below 0.6, so the task cannot
degenerate into intensity thresholding as the generator evolves.

Augmentation is restricted to flips and 90° rotations — exactly label-preserving
and interpolation-free, so masks stay binary. Arbitrary-angle rotation would blur
mask edges into fractional values.

## Using real data

Replace `build_datasets` with a loader yielding `(1, H, W)` image and mask
tensors. For NIfTI or DICOM, uncomment `nibabel` / `pydicom` in
`requirements.txt`. Good public sets: [Medical Segmentation
Decathlon](http://medicaldecathlon.com/), [BraTS](https://www.synapse.org/brats),
[ISIC](https://challenge.isic-archive.com/) for skin lesions,
[LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri) for lung
nodules.

For 3D volumes, swap `Conv2d`/`MaxPool2d` for their 3D equivalents — the
architecture, losses and metrics all carry over unchanged.

## Layout

```
src/
  data.py       synthetic scans with bias field, texture, empty cases
  model.py      U-Net, skip connections, TTA
  losses.py     Dice, Tversky, Focal, combined, deep supervision
  metrics.py    Dice, IoU, HD95, surface distance, case-level outcomes,
                connected-component filtering
  train.py      training loop, threshold tuning, post-processing comparison
  predict.py    inference and overlay rendering
tests/          pytest suite
```

## License

MIT
