# Medical Image Segmentation with U-Net

[![tests](https://github.com/nature123-web/medical-image-segmentation/actions/workflows/tests.yml/badge.svg)](https://github.com/nature123-web/medical-image-segmentation/actions/workflows/tests.yml)

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

pytest    # 101 tests
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
| **Boundary** | distance-weighted surface loss — the only one that knows *how far* a wrong pixel is from the truth |

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

### The boundary loss covers a real blind spot

Dice and cross-entropy are *regional*: they count pixels. A false positive
touching the lesion and one on the opposite side of the image cost exactly the
same — even though only the second is clinically alarming. That is precisely the
gap HD95 measures, and until now this repo reported it without optimising it.

`BoundaryLoss` integrates the prediction against the ground truth's **signed
distance field**, so error is weighted by distance. Enable with
`loss.boundary_weight`. Two details that matter:

- It is **ramped in** over `boundary_warmup_epochs`. At full strength from step 0
  the distance term dominates while the prediction is still far from the truth,
  and training destabilises before Dice has found the lesion at all.
- It is used **alongside** Dice, never alone — on its own the empty prediction
  is a valid minimiser, since it has no notion of overlap.

Distances are normalised by the image diagonal so `boundary_weight` means the
same thing at 128×128 and 512×512.

## Attention gates

`model.attention: true` switches on additive attention gates (Oktay et al.,
2018). The plain U-Net concatenates the entire encoder feature map into the
decoder, background included — and when the foreground is under 1% of the image,
that is overwhelmingly irrelevant signal the decoder must learn to ignore.

The gate computes a spatial attention map from the *coarser* decoder features,
which already know roughly where the object is, and multiplies the skip by it
before concatenation. Regions the decoder has no interest in are suppressed at
source.

The maps are inspectable, which is worth as much as the accuracy:

```python
logits, attention_maps = model(images, return_attention=True)
# one (B, 1, H, W) map per decoder level, coarsest first
```

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

## Uncertainty: segment confidently, flag the rest

A binary mask says *what* the model decided and nothing about how sure it was —
the wrong output for a tool sitting in front of a radiologist. The useful
behaviour is to segment where it can and flag where it cannot, so review time
goes to the ambiguous cases.

`src/uncertainty.py` keeps dropout active at inference and samples several
forward passes (Gal & Ghahramani, 2016), then splits the result:

- **aleatoric** — inherent ambiguity in the image, like a genuinely fuzzy
  boundary. More training data will not reduce it.
- **epistemic** — the model's own ignorance, like an unusual presentation. More
  data *will*.

That split is what tells you which cases are worth labelling and which are just
hard. Measured on a held-out set from a short real training run:

```
             boundary band   far background    ratio
total            0.3113          0.1660          1.9x
epistemic        0.0261          0.0023         11.3x
aleatoric        0.2852          0.1637          1.7x

uncertainty-vs-error correlation: 0.791
```

**Epistemic uncertainty localises 11× better than total.** The aleatoric
component is diffuse — image noise is everywhere — and swamps the signal if you
only look at the total, which is the case for reporting the decomposition rather
than a single number.

The last line is the claim any uncertainty estimate has to earn: cases the model
is unsure about are the cases it gets wrong. At 0.79 Spearman, routing review by
uncertainty is substantially better than routing at random. A correlation near
zero would mean the uncertainty is noise, and `uncertainty_error_correlation`
exists to check exactly that.

```python
out = mc_dropout_predict(model, images, n_samples=20)
out["mean"]        # the prediction to threshold
out["epistemic"]   # where the model is out of its depth
review = rank_cases_for_review(case_uncertainties, budget=0.1)
```

One implementation detail worth stating, because it is the standard mistake:
`enable_dropout` activates **only** the dropout layers. Calling `model.train()`
would also reactivate BatchNorm's batch statistics, so predictions would depend
on whatever else shared the batch and the running estimates would drift. There
is a test for it.

Requires `model.dropout > 0` at training time — with dropout at 0 every sample is
identical and the uncertainty map is meaningless rather than merely small, so
the function says so out loud instead of returning a silent zero map.

## Inference on images larger than the training crop

A U-Net trained on 128×128 patches cannot be handed a 2048×2048 slide: memory is
quadratic in the side length, and the network has never seen that field of view.
`sliding_window_inference` tiles the image with overlapping patches and blends.

The blending is the part that goes wrong. Averaging patches uniformly produces
visible **seams**, because a pixel at a patch edge was predicted with almost no
surrounding context and is systematically less reliable than one at the centre.
Weighting each patch by a centred Gaussian and normalising by the accumulated
weight makes the confident middle of one patch dominate the unreliable border of
its neighbour — and makes the scheme an exact partition of unity.

That exactness is testable, and tested: an identity model must reconstruct its
input *exactly*, at any overlap and any image size, including sizes that are not
a whole number of strides.

```bash
python -m src.predict --checkpoint runs/base/best.pt --patch-size 128 --overlap 0.5
```

Probabilities are blended, not logits — logits from different patches are not on
a comparable scale, so averaging them lets one overconfident patch dominate.

## Post-processing and thresholding

**Threshold tuning.** 0.5 is only optimal for balanced classes. At ~1% foreground
the model is systematically under-confident and the best operating point is well
below 0.5. The run tunes it on validation and reports both.

**Small-component removal.** A model whose output is scattered with 3-pixel
specks can have good Dice and be unusable — every speck is an alert someone must
dismiss. Connected-component filtering is applied and the effect reported
separately, so you can see what it bought.

Every run prints all three stages. From an actual (deliberately small) run:

```
U-Net                              dice 0.4805   hd95 10.9139
U-Net + flip TTA                   dice 0.5936   hd95  5.4003
U-Net + TTA + components >= 20px   dice 0.5811   hd95  4.7006
```

Note the last row: component filtering **lowers Dice** (0.594 → 0.581) while
improving the boundary metric (5.40 → 4.70). It removes small true-positive
fragments along with the specks. Whether that trade is worth taking depends on
whether you are optimising a leaderboard or a radiologist's attention, which is
exactly why all three are reported rather than only the best one.

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
  data.py         synthetic scans with bias field, texture, empty cases
  model.py        U-Net, skip connections, attention gates, TTA
  losses.py       Dice, Tversky, Focal, boundary, combined, deep supervision
  metrics.py      Dice, IoU, HD95, surface distance, case-level outcomes,
                  connected-component filtering
  inference.py    Gaussian-weighted sliding window for large images
  uncertainty.py  MC dropout, aleatoric/epistemic split, review ranking
  train.py        training loop, threshold tuning, post-processing comparison
  predict.py      inference and overlay rendering
tests/            pytest suite (101 tests)
```

## License

MIT
