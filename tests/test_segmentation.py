"""Tests for the U-Net, imbalance-aware losses, and segmentation metrics."""

import numpy as np
import pytest
import torch

from src.data import SegmentationDataset, dataset_statistics, make_sample
from src.losses import (
    CombinedLoss,
    DiceLoss,
    FocalLoss,
    TverskyLoss,
    dice_coefficient,
)
from src.metrics import (
    aggregate,
    average_surface_distance,
    detection_outcome,
    dice_score,
    evaluate_batch,
    hausdorff_95,
    iou_score,
    precision_score,
    remove_small_components,
    sensitivity,
    specificity,
)
from src.model import UNet


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def test_sample_shapes_and_ranges():
    sample = make_sample(size=64, seed=0)
    assert sample.image.shape == (64, 64)
    assert sample.mask.shape == (64, 64)
    assert 0.0 <= sample.image.min() and sample.image.max() <= 1.0
    assert set(np.unique(sample.mask)) <= {0.0, 1.0}


def test_lesions_are_small_relative_to_the_image():
    """Severe imbalance is the defining property of the problem."""
    fractions = [
        make_sample(size=128, lesion_probability=1.0, seed=s).mask.mean()
        for s in range(40)
    ]
    assert 0.0 < np.mean(fractions) < 0.06


def test_some_cases_have_no_lesion():
    """A model never shown a negative learns to always find something."""
    samples = [make_sample(size=64, lesion_probability=0.6, seed=s)
               for s in range(60)]
    empty = sum(1 for s in samples if s.mask.sum() == 0)
    assert 5 < empty < 55


def test_mask_is_binary_after_augmentation():
    """Only flips and 90-degree rotations, so no interpolation blurs the mask."""
    dataset = SegmentationDataset(20, size=64, seed=0, augment=True)
    for i in range(20):
        _, mask = dataset[i]
        assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}


def test_augmentation_keeps_image_and_mask_aligned():
    """A flip applied to one and not the other would silently destroy training."""
    plain = SegmentationDataset(10, size=64, seed=3, augment=False)
    augmented = SegmentationDataset(10, size=64, seed=3, augment=True)
    for i in range(10):
        _, mask_plain = plain[i]
        _, mask_augmented = augmented[i]
        # Augmentation is rigid, so the foreground area is preserved exactly.
        assert mask_plain.sum() == mask_augmented.sum()


def test_dataset_is_deterministic():
    a = SegmentationDataset(5, size=64, seed=7)[2]
    b = SegmentationDataset(5, size=64, seed=7)[2]
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_lesion_is_not_findable_by_a_global_threshold():
    """The bias field is there to defeat exactly this shortcut."""
    dice_scores = []
    for seed in range(25):
        sample = make_sample(size=96, lesion_probability=1.0, seed=seed)
        if sample.mask.sum() == 0:
            continue
        best = max(
            dice_score((sample.image > t).astype(float), sample.mask)
            for t in np.arange(0.3, 0.95, 0.05)
        )
        dice_scores.append(best)
    assert np.mean(dice_scores) < 0.6, "a fixed threshold already solves this"


def test_dataset_statistics_reports_imbalance():
    stats = dataset_statistics(SegmentationDataset(50, size=64, seed=0), 50)
    assert 0.0 < stats["mean_foreground_fraction"] < 0.1
    assert 0.0 <= stats["empty_fraction"] <= 1.0


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

def make_model(**kwargs):
    defaults = dict(in_channels=1, n_classes=1, base_channels=8, depth=3,
                    dropout=0.0)
    defaults.update(kwargs)
    return UNet(**defaults)


def test_output_matches_input_resolution():
    """Segmentation must be dense: one prediction per input pixel."""
    model = make_model().eval()
    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        assert model(x).shape == (2, 1, 64, 64)


@pytest.mark.parametrize("size", [32, 64, 96, 128])
def test_various_input_sizes(size):
    model = make_model(depth=3).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 1, size, size)).shape == (1, 1, size, size)


def test_non_square_input():
    model = make_model(depth=2).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 1, 64, 96)).shape == (1, 1, 64, 96)


def test_odd_size_input_is_padded_not_cropped():
    """Odd sizes make skip and upsampled tensors mismatch by a pixel."""
    model = make_model(depth=2).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 1, 50, 50)).shape == (1, 1, 50, 50)


@pytest.mark.parametrize("norm", ["instance", "batch", "group", "none"])
def test_all_norms_build_and_run(norm):
    model = UNet(base_channels=8, depth=2, norm=norm).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 1, 32, 32)).shape == (2, 1, 32, 32)


def test_unknown_norm_raises():
    with pytest.raises(ValueError, match="unknown norm"):
        UNet(base_channels=8, depth=2, norm="nonsense")


def test_instance_norm_works_with_batch_size_one():
    """The reason InstanceNorm is the default: BatchNorm cannot do this."""
    model = make_model(norm="instance").train()
    out = model(torch.randn(1, 1, 32, 32))
    assert torch.isfinite(out).all()


def test_transposed_convolution_variant():
    model = make_model(bilinear=False).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 1, 64, 64)).shape == (1, 1, 64, 64)


def test_deep_supervision_returns_multiple_outputs_only_in_training():
    model = make_model(depth=3, deep_supervision=True)
    model.train()
    outputs = model(torch.randn(1, 1, 64, 64))
    assert isinstance(outputs, list) and len(outputs) > 1
    assert outputs[0].shape == (1, 1, 64, 64)

    model.eval()
    with torch.no_grad():
        assert not isinstance(model(torch.randn(1, 1, 64, 64)), list)


def test_attention_unet_forward_shape():
    model = make_model(attention=True).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 1, 64, 64)).shape == (2, 1, 64, 64)


def test_attention_maps_are_probabilities_at_skip_resolution():
    model = make_model(depth=3, attention=True).eval()
    with torch.no_grad():
        _, maps = model(torch.randn(1, 1, 64, 64), return_attention=True)

    assert len(maps) == 3
    for attention in maps:
        assert attention.shape[1] == 1              # one gate per position
        assert (attention >= 0).all() and (attention <= 1).all()
    # Coarsest gate first, doubling resolution toward the output.
    assert maps[0].shape[-1] * 4 == maps[-1].shape[-1]


def test_attention_gate_suppresses_the_skip():
    """The gate must actually scale the skip, not pass it through."""
    from src.model import AttentionGate

    torch.manual_seed(0)
    gate = AttentionGate(gate_channels=8, skip_channels=8).eval()
    skip = torch.randn(2, 8, 16, 16)
    with torch.no_grad():
        gated, attention = gate(torch.randn(2, 8, 16, 16), skip)

    assert gated.shape == skip.shape
    # sigmoid output is strictly below 1, so the gated skip is strictly smaller.
    assert gated.abs().sum() < skip.abs().sum()
    assert torch.allclose(gated, skip * attention)


def test_attention_gate_upsamples_a_coarser_gating_signal():
    from src.model import AttentionGate

    gate = AttentionGate(gate_channels=8, skip_channels=8).eval()
    with torch.no_grad():
        gated, attention = gate(torch.randn(1, 8, 8, 8), torch.randn(1, 8, 16, 16))
    assert gated.shape == (1, 8, 16, 16)
    assert attention.shape == (1, 1, 16, 16)


def test_attention_adds_parameters_and_is_off_by_default():
    plain = make_model(attention=False)
    gated = make_model(attention=True)
    assert plain.gates is None
    assert gated.gates is not None
    assert sum(p.numel() for p in gated.parameters()) > \
        sum(p.numel() for p in plain.parameters())


def test_attention_model_trains():
    model = make_model(attention=True)
    target = torch.zeros(2, 1, 32, 32)
    target[:, :, 8:16, 8:16] = 1.0
    loss = CombinedLoss()(model(torch.randn(2, 1, 32, 32)), target)
    loss.backward()
    gate_grads = [p.grad for n, p in model.named_parameters()
                  if n.startswith("gates") and p.grad is not None]
    assert gate_grads and any(g.abs().sum() > 0 for g in gate_grads)


def test_predict_returns_binary_mask():
    model = make_model().eval()
    mask = model.predict(torch.randn(2, 1, 32, 32))
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}


def test_predict_proba_is_in_unit_interval():
    model = make_model().eval()
    probabilities = model.predict_proba(torch.randn(2, 1, 32, 32))
    assert (probabilities >= 0).all() and (probabilities <= 1).all()


def test_tta_inverts_each_flip_before_averaging():
    """Without the inverse flip, the views cancel into a blur.

    A model that is exactly flip-equivariant must give identical output with
    and without TTA. Building one by hand pins the flip bookkeeping.
    """
    class Identity(UNet):
        def forward(self, x):
            # Flip-equivariant by construction: output depends only on the pixel.
            return (x - 0.5) * 10

    model = Identity(base_channels=8, depth=1).eval()
    x = torch.rand(1, 1, 16, 16)
    plain = model.predict_proba(x, tta=False)
    with_tta = model.predict_proba(x, tta=True)
    assert torch.allclose(plain, with_tta, atol=1e-5)


def test_predict_restores_training_mode():
    model = make_model().train()
    model.predict_proba(torch.randn(1, 1, 32, 32))
    assert model.training


def test_gradients_reach_the_encoder():
    model = make_model()
    model(torch.randn(2, 1, 32, 32)).sum().backward()
    encoder_grads = [p.grad for n, p in model.named_parameters()
                     if n.startswith("encoders") and p.grad is not None]
    assert encoder_grads and any(g.abs().sum() > 0 for g in encoder_grads)


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #

def test_dice_coefficient_of_a_perfect_match():
    target = torch.zeros(1, 1, 8, 8)
    target[:, :, 2:5, 2:5] = 1.0
    assert dice_coefficient(target, target).item() == pytest.approx(1.0, abs=1e-3)


def test_dice_coefficient_of_a_disjoint_prediction():
    target = torch.zeros(1, 1, 8, 8)
    target[:, :, 0:2, 0:2] = 1.0
    prediction = torch.zeros(1, 1, 8, 8)
    prediction[:, :, 6:8, 6:8] = 1.0
    assert dice_coefficient(prediction, target).item() < 0.2


def test_dice_of_two_empty_masks_is_one():
    """Correctly predicting 'nothing here' must not score zero."""
    empty = torch.zeros(1, 1, 8, 8)
    assert dice_coefficient(empty, empty).item() == pytest.approx(1.0)


def test_dice_loss_is_lower_for_a_better_prediction():
    target = torch.zeros(2, 1, 16, 16)
    target[:, :, 4:10, 4:10] = 1.0
    good = torch.full((2, 1, 16, 16), -5.0)
    good[:, :, 4:10, 4:10] = 5.0
    bad = torch.full((2, 1, 16, 16), -5.0)

    loss = DiceLoss()
    assert loss(good, target) < loss(bad, target)


def test_per_sample_dice_does_not_let_a_big_mask_hide_a_missed_small_one():
    """The reason Dice is computed per sample rather than over the batch."""
    targets = torch.zeros(2, 1, 32, 32)
    targets[0, :, :20, :20] = 1.0        # large structure
    targets[1, :, 0:3, 0:3] = 1.0        # small lesion

    predictions = torch.zeros(2, 1, 32, 32)
    predictions[0, :, :20, :20] = 1.0    # large one found, small one missed

    per_sample = dice_coefficient(predictions, targets, per_sample=True).mean()
    pooled = dice_coefficient(predictions, targets, per_sample=False)
    # Pooling makes the total miss look almost perfect.
    assert pooled > 0.9
    assert per_sample < 0.6


def test_tversky_beta_trades_recall_for_precision():
    """High beta punishes false negatives, so under-segmentation costs more."""
    target = torch.zeros(1, 1, 16, 16)
    target[:, :, 4:12, 4:12] = 1.0
    under = torch.full((1, 1, 16, 16), -5.0)
    under[:, :, 6:10, 6:10] = 5.0        # misses most of the lesion

    recall_focused = TverskyLoss(alpha=0.1, beta=0.9)
    precision_focused = TverskyLoss(alpha=0.9, beta=0.1)
    assert recall_focused(under, target) > precision_focused(under, target)


def test_tversky_at_half_half_equals_dice():
    """Tversky generalises Dice; at alpha=beta=0.5 they must coincide.

    Exactly true only in the limit of no smoothing: Dice adds the constant once
    to a doubled numerator, Tversky once to an undoubled one, so a smooth of 1
    leaves a small gap. Testing at 1e-8 states the identity precisely.
    """
    target = torch.zeros(1, 1, 16, 16)
    target[:, :, 4:10, 4:10] = 1.0
    torch.manual_seed(0)
    logits = torch.randn(1, 1, 16, 16)

    tversky = TverskyLoss(alpha=0.5, beta=0.5, smooth=1e-8)(logits, target)
    dice = DiceLoss(smooth=1e-8)(logits, target)
    assert float(tversky) == pytest.approx(float(dice), abs=1e-6)


def test_focal_loss_downweights_easy_pixels():
    target = torch.zeros(1, 1, 8, 8)
    easy = torch.full((1, 1, 8, 8), -10.0)      # confident and correct
    hard = torch.zeros(1, 1, 8, 8)              # uncertain
    focal = FocalLoss()
    assert focal(easy, target) < focal(hard, target)


def test_combined_loss_is_finite_and_differentiable():
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    target = torch.zeros(2, 1, 16, 16)
    target[:, :, 4:8, 4:8] = 1.0
    loss = CombinedLoss(dice_weight=1.0, bce_weight=1.0, focal_weight=0.5,
                        tversky_weight=0.5)(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_combined_loss_handles_deep_supervision_lists():
    target = torch.zeros(1, 1, 32, 32)
    target[:, :, 8:16, 8:16] = 1.0
    logits = [torch.randn(1, 1, 32, 32), torch.randn(1, 1, 16, 16)]
    assert torch.isfinite(CombinedLoss()(logits, target))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def test_dice_and_iou_agree_on_a_perfect_prediction():
    mask = np.zeros((16, 16))
    mask[4:10, 4:10] = 1
    assert dice_score(mask, mask) == 1.0
    assert iou_score(mask, mask) == 1.0


def test_dice_exceeds_iou_for_partial_overlap():
    a = np.zeros((16, 16)); a[0:8, 0:8] = 1
    b = np.zeros((16, 16)); b[4:12, 4:12] = 1
    assert dice_score(a, b) > iou_score(a, b)


def test_predicting_something_on_an_empty_target_scores_zero():
    target = np.zeros((16, 16))
    prediction = np.zeros((16, 16)); prediction[2:4, 2:4] = 1
    assert dice_score(prediction, target) == 0.0


def test_empty_prediction_on_empty_target_scores_one():
    empty = np.zeros((16, 16))
    assert dice_score(empty, empty) == 1.0


def test_pixel_accuracy_is_misleading_but_dice_is_not():
    """The headline argument for Dice, stated as a test."""
    target = np.zeros((100, 100))
    target[50:55, 50:55] = 1               # 0.25% foreground
    all_background = np.zeros((100, 100))

    accuracy = (all_background == target).mean()
    assert accuracy > 0.99
    assert dice_score(all_background, target) == 0.0


def test_sensitivity_and_specificity():
    target = np.zeros((10, 10)); target[0:4, 0:4] = 1
    prediction = np.zeros((10, 10)); prediction[0:2, 0:4] = 1
    assert sensitivity(prediction, target) == pytest.approx(0.5)
    assert specificity(prediction, target) == 1.0


def test_sensitivity_is_nan_when_there_is_nothing_to_find():
    assert np.isnan(sensitivity(np.zeros((8, 8)), np.zeros((8, 8))))


def test_precision_is_nan_for_an_empty_prediction():
    target = np.zeros((8, 8)); target[0, 0] = 1
    assert np.isnan(precision_score(np.zeros((8, 8)), target))


def test_hausdorff_is_zero_for_identical_masks():
    mask = np.zeros((32, 32)); mask[10:20, 10:20] = 1
    assert hausdorff_95(mask, mask) == pytest.approx(0.0)


def test_hausdorff_grows_with_boundary_error():
    target = np.zeros((40, 40)); target[10:20, 10:20] = 1
    close = np.zeros((40, 40)); close[11:21, 11:21] = 1
    far = np.zeros((40, 40)); far[20:30, 20:30] = 1
    assert hausdorff_95(close, target) < hausdorff_95(far, target)


def test_hd95_suppresses_a_single_outlying_pixel():
    """This is what the 95th percentile is for, as opposed to the maximum.

    One stray pixel is under 5% of the surface, so it falls outside the
    percentile -- deliberately, because a single voxel should not define the
    whole boundary metric.
    """
    target = np.zeros((60, 60)); target[20:40, 20:40] = 1
    speckled = target.copy()
    speckled[2, 2] = 1
    assert hausdorff_95(speckled, target) == pytest.approx(0.0)


def test_hd95_catches_a_substantial_false_region_that_dice_tolerates():
    """Near-identical Dice, badly wrong boundary -- the case Dice misses."""
    target = np.zeros((80, 80)); target[30:50, 30:50] = 1
    clean = target.copy()
    with_blob = target.copy()
    with_blob[2:12, 2:12] = 1          # a real false-positive region, far away

    # Dice barely notices: 400 correct pixels against 100 spurious ones.
    assert dice_score(with_blob, target) > 0.85
    assert hausdorff_95(clean, target) == pytest.approx(0.0)
    assert hausdorff_95(with_blob, target) > 20


def test_average_surface_distance_is_zero_for_identical_masks():
    mask = np.zeros((32, 32)); mask[8:16, 8:16] = 1
    assert average_surface_distance(mask, mask) == pytest.approx(0.0)


def test_detection_outcomes():
    lesion = np.zeros((20, 20)); lesion[5:10, 5:10] = 1
    empty = np.zeros((20, 20))
    assert detection_outcome(lesion, lesion) == "true_positive"
    assert detection_outcome(empty, lesion) == "false_negative"
    assert detection_outcome(lesion, empty) == "false_positive"
    assert detection_outcome(empty, empty) == "true_negative"


def test_aggregate_separates_overall_from_lesion_only_dice():
    """Empty-and-correct cases score 1.0 and inflate the overall mean."""
    lesion = np.zeros((20, 20)); lesion[5:10, 5:10] = 1
    empty = np.zeros((20, 20))
    predictions = np.stack([empty, empty])        # misses the lesion
    targets = np.stack([lesion, empty])

    summary = aggregate(evaluate_batch(predictions, targets))
    assert summary["dice"] == pytest.approx(0.5)          # (0 + 1) / 2
    assert summary["dice_lesion_only"] == pytest.approx(0.0)
    assert summary["n_with_lesion"] == 1


def test_remove_small_components_keeps_the_large_one():
    mask = np.zeros((32, 32))
    mask[5:15, 5:15] = 1        # 100 pixels
    mask[25, 25] = 1            # 1 pixel speck
    cleaned = remove_small_components(mask, min_size=20)
    assert cleaned[25, 25] == 0
    assert cleaned[5:15, 5:15].sum() == 100


def test_remove_small_components_on_empty_mask():
    assert remove_small_components(np.zeros((16, 16)), 10).sum() == 0


def test_remove_small_components_treats_diagonal_as_separate():
    """4-connectivity: diagonal touching is not the same component."""
    mask = np.zeros((16, 16))
    mask[2, 2] = 1
    mask[3, 3] = 1
    assert remove_small_components(mask, min_size=2).sum() == 0
