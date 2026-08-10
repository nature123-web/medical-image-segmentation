"""Segmentation metrics, including the boundary ones that overlap misses.

Pixel accuracy is meaningless here. On an image where the lesion is 1% of the
pixels, predicting all-background scores 99% accuracy and Dice 0. Everything in
this module is chosen to be sensitive to the foreground.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def dice_score(prediction: np.ndarray, target: np.ndarray,
               empty_score: float = 1.0) -> float:
    """2|A∩B| / (|A|+|B|).

    ``empty_score`` is returned when both masks are empty -- a correct
    prediction of "no lesion here", which is the right answer on a negative
    case and must not be scored 0. Predicting something on an empty target
    still scores 0, as it should.
    """
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    total = prediction.sum() + target.sum()
    if total == 0:
        return empty_score
    return float(2.0 * np.logical_and(prediction, target).sum() / total)


def iou_score(prediction: np.ndarray, target: np.ndarray,
              empty_score: float = 1.0) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    union = np.logical_or(prediction, target).sum()
    if union == 0:
        return empty_score
    return float(np.logical_and(prediction, target).sum() / union)


def sensitivity(prediction: np.ndarray, target: np.ndarray) -> float:
    """Recall on the foreground. nan when there is nothing to find."""
    target = target.astype(bool)
    if target.sum() == 0:
        return float("nan")
    return float(np.logical_and(prediction.astype(bool), target).sum()
                 / target.sum())


def specificity(prediction: np.ndarray, target: np.ndarray) -> float:
    background = ~target.astype(bool)
    if background.sum() == 0:
        return float("nan")
    return float(np.logical_and(~prediction.astype(bool), background).sum()
                 / background.sum())


def precision_score(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction.astype(bool)
    if prediction.sum() == 0:
        return float("nan")
    return float(np.logical_and(prediction, target.astype(bool)).sum()
                 / prediction.sum())


def _surface_points(mask: np.ndarray) -> np.ndarray:
    """Coordinates of boundary pixels: foreground with a background neighbour."""
    mask = mask.astype(bool)
    if not mask.any():
        return np.empty((0, 2))
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    interior = (padded[:-2, 1:-1] & padded[2:, 1:-1]
                & padded[1:-1, :-2] & padded[1:-1, 2:])
    return np.argwhere(mask & ~interior)


def hausdorff_95(prediction: np.ndarray, target: np.ndarray) -> float:
    """95th-percentile symmetric surface distance, in pixels.

    Reported alongside Dice because they measure different failures. A
    prediction can have excellent Dice and a terrible boundary -- one stray
    voxel far from the lesion barely moves Dice but is exactly what a clinician
    notices. The 95th percentile rather than the maximum keeps a single
    outlying pixel from defining the whole metric.
    """
    a = _surface_points(prediction)
    b = _surface_points(target)
    if len(a) == 0 and len(b) == 0:
        return 0.0
    if len(a) == 0 or len(b) == 0:
        return float("nan")     # undefined, not infinitely bad

    distances = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    forward = distances.min(axis=1)
    backward = distances.min(axis=0)
    return float(max(np.percentile(forward, 95), np.percentile(backward, 95)))


def average_surface_distance(prediction: np.ndarray, target: np.ndarray
                             ) -> float:
    a = _surface_points(prediction)
    b = _surface_points(target)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    distances = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    return float((distances.min(axis=1).mean() + distances.min(axis=0).mean()) / 2)


def detection_outcome(prediction: np.ndarray, target: np.ndarray,
                      min_overlap: float = 0.1) -> str:
    """Case-level outcome, which is how screening tools are actually judged.

    A radiologist cares first about "did it find the lesion at all", not about
    the exact boundary. Returns one of ``true_positive``, ``false_positive``,
    ``true_negative``, ``false_negative``.
    """
    has_target = target.astype(bool).any()
    has_prediction = prediction.astype(bool).any()
    if not has_target:
        return "false_positive" if has_prediction else "true_negative"
    if not has_prediction:
        return "false_negative"
    return ("true_positive" if dice_score(prediction, target) >= min_overlap
            else "false_negative")


def evaluate_batch(predictions: np.ndarray, targets: np.ndarray,
                   compute_boundary: bool = True) -> List[Dict[str, float]]:
    """Per-image metrics for a batch of (N, H, W) masks."""
    results = []
    for prediction, target in zip(predictions, targets):
        entry = {
            "dice": dice_score(prediction, target),
            "iou": iou_score(prediction, target),
            "sensitivity": sensitivity(prediction, target),
            "specificity": specificity(prediction, target),
            "precision": precision_score(prediction, target),
            "outcome": detection_outcome(prediction, target),
            "has_lesion": bool(target.astype(bool).any()),
        }
        if compute_boundary:
            entry["hd95"] = hausdorff_95(prediction, target)
            entry["asd"] = average_surface_distance(prediction, target)
        results.append(entry)
    return results


def aggregate(results: List[Dict[str, float]]) -> Dict[str, float]:
    """Summarise per-image results.

    Dice is reported twice: over all cases, and over lesion-bearing cases only.
    The two differ a lot when the dataset has negatives, because empty-and-
    correct scores 1.0 and inflates the overall mean. Both numbers are needed --
    the first says how the tool behaves in a screening cohort, the second says
    how well it actually segments.
    """
    if not results:
        return {}

    def mean_of(key, subset=None):
        values = [r[key] for r in (subset or results)
                  if isinstance(r.get(key), float) and r[key] == r[key]]
        return float(np.mean(values)) if values else float("nan")

    with_lesion = [r for r in results if r["has_lesion"]]
    counts = {name: sum(1 for r in results if r["outcome"] == name)
              for name in ("true_positive", "false_positive",
                           "true_negative", "false_negative")}

    detected = counts["true_positive"] + counts["false_negative"]
    flagged = counts["true_positive"] + counts["false_positive"]
    return {
        "dice": mean_of("dice"),
        "dice_lesion_only": mean_of("dice", with_lesion),
        "iou": mean_of("iou"),
        "sensitivity": mean_of("sensitivity"),
        "specificity": mean_of("specificity"),
        "precision": mean_of("precision"),
        "hd95": mean_of("hd95"),
        "asd": mean_of("asd"),
        "case_sensitivity": (counts["true_positive"] / detected
                             if detected else float("nan")),
        "case_precision": (counts["true_positive"] / flagged
                           if flagged else float("nan")),
        "n_cases": len(results),
        "n_with_lesion": len(with_lesion),
        **counts,
    }


def format_report(summary: Dict[str, float], name: str = "model") -> str:
    lines = [f"{name}:"]
    for key in ("dice", "dice_lesion_only", "iou", "sensitivity", "precision",
                "specificity", "hd95", "asd", "case_sensitivity",
                "case_precision"):
        if key in summary:
            lines.append(f"  {key:<20} {summary[key]:.4f}")
    lines.append(
        f"  cases: {summary.get('n_cases', 0)} "
        f"({summary.get('n_with_lesion', 0)} with lesion)  "
        f"TP={summary.get('true_positive', 0)} "
        f"FP={summary.get('false_positive', 0)} "
        f"TN={summary.get('true_negative', 0)} "
        f"FN={summary.get('false_negative', 0)}"
    )
    return "\n".join(lines)


def remove_small_components(mask: np.ndarray, min_size: int = 20) -> np.ndarray:
    """Drop connected components below ``min_size`` pixels.

    Standard post-processing: a segmentation model scattered with 3-pixel
    specks has good Dice and is unusable, because every speck is an alert
    someone must dismiss. Implemented with an iterative flood fill so scipy is
    not required.
    """
    mask = mask.astype(bool)
    if not mask.any():
        return mask.astype(np.float32)

    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    current, sizes = 0, {}

    for start_y in range(height):
        for start_x in range(width):
            if not mask[start_y, start_x] or labels[start_y, start_x]:
                continue
            current += 1
            stack = [(start_y, start_x)]
            labels[start_y, start_x] = current
            size = 0
            while stack:
                y, x = stack.pop()
                size += 1
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < height and 0 <= nx < width
                            and mask[ny, nx] and not labels[ny, nx]):
                        labels[ny, nx] = current
                        stack.append((ny, nx))
            sizes[current] = size

    keep = {label for label, size in sizes.items() if size >= min_size}
    return np.isin(labels, list(keep)).astype(np.float32)
