"""Shared, deterministic classical image segmentation utilities.

No deep model, reference mask, SAM prompt or ground-truth box is used here.
Only the original BGR image and an explicit configuration are accepted.

Developed with Arena.ai assistance; review and disclose its use.
"""

from pathlib import Path
import hashlib
import math

import cv2
import numpy as np


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resize_for_inference(image, max_side):
    height, width = image.shape[:2]
    scale = min(1.0, float(max_side) / max(height, width))
    if scale == 1.0:
        return image, scale
    small = cv2.resize(image, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA)
    return small, scale


def odd_kernel(value):
    value = max(0, int(value))
    if value == 0:
        return None
    if value % 2 == 0:
        value += 1
    return np.ones((value, value), np.uint8)


def morphology(mask, open_size=3, close_size=5):
    mask = (mask > 0).astype(np.uint8) * 255
    opening, closing = odd_kernel(open_size), odd_kernel(close_size)
    if opening is not None:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opening)
    if closing is not None:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing)
    return mask


def select_components(mask, min_area_frac=0.001, max_area_frac=0.65,
                      max_components=4, center_weight=1.0):
    """Filter connected components using only area, shape and image geometry."""
    height, width = mask.shape
    image_area = height * width
    count, labels, stats, centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    candidates = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        fraction = float(area) / image_area
        if fraction < min_area_frac or fraction > max_area_frac or w < 4 or h < 4:
            continue
        cx, cy = centroids[label]
        distance = math.hypot((cx - width / 2) / width, (cy - height / 2) / height)
        fill = float(area) / max(1, w * h)
        aspect = min(w, h) / max(w, h)
        # Larger, compact and more central components rank higher. No labels used.
        score = (fraction ** 0.35) * (0.5 + 0.5 * fill) * (0.7 + 0.3 * aspect) / (1 + center_weight * distance)
        candidates.append((score, label))
    candidates.sort(reverse=True)
    output = np.zeros_like(mask, dtype=np.uint8)
    for _, label in candidates[:max(1, int(max_components))]:
        output[labels == label] = 255
    return output


def hsv_seed(image, config):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    pale = (saturation <= config["sat_max"]) & (value >= config["val_min"])
    if config.get("warm", False):
        warm = (hue >= config.get("warm_hue_min", 3)) & (hue <= config.get("warm_hue_max", 35))
        warm &= (saturation <= config.get("warm_sat_max", 180)) & (value >= config["val_min"])
        pale |= warm
    return (pale.astype(np.uint8) * 255)


def lab_center_seed(image, config):
    """Appearance deviation from an automatically sampled central image patch."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    height, width = lab.shape[:2]
    patch = config.get("center_patch", 0.16)
    half_w, half_h = width * patch / 2, height * patch / 2
    cx, cy = width / 2, height / 2
    sample = lab[int(cy-half_h):int(cy+half_h)+1, int(cx-half_w):int(cx+half_w)+1]
    if sample.size == 0:
        return np.zeros((height, width), np.uint8)
    median = np.median(sample.reshape(-1, 3), axis=0)
    scales = np.array(config.get("lab_scales", [45.0, 22.0, 22.0]), dtype=np.float32)
    distance = np.sqrt(np.sum(((lab - median) / scales) ** 2, axis=2))
    return ((distance <= config["lab_threshold"]).astype(np.uint8) * 255)


def grabcut_rect(image, config):
    height, width = image.shape[:2]
    scale = config["rect_scale"]
    w, h = max(4, int(width * scale)), max(4, int(height * scale))
    x, y = (width - w) // 2, (height - h) // 2
    mask = np.zeros((height, width), np.uint8)
    background, foreground = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(image, mask, (x, y, w, h), background, foreground,
                    int(config.get("grabcut_iters", 4)), cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return np.zeros((height, width), np.uint8)
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def refine_with_grabcut(image, seed, config):
    """Classical GrabCut refinement from an image-derived seed mask."""
    height, width = seed.shape
    if cv2.countNonZero(seed) < 20:
        return seed
    erosion = odd_kernel(config.get("sure_fg_kernel", 9))
    dilation = odd_kernel(config.get("sure_bg_kernel", 15))
    sure_fg = cv2.erode(seed, erosion) if erosion is not None else seed
    expanded = cv2.dilate(seed, dilation) if dilation is not None else seed
    gc = np.full((height, width), cv2.GC_PR_BGD, np.uint8)
    gc[expanded == 0] = cv2.GC_BGD
    gc[seed > 0] = cv2.GC_PR_FGD
    gc[sure_fg > 0] = cv2.GC_FGD
    if cv2.countNonZero(sure_fg) < 10 or cv2.countNonZero(gc == cv2.GC_BGD) < 10:
        return seed
    background, foreground = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(image, gc, None, background, foreground,
                    int(config.get("grabcut_iters", 4)), cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return seed
    return np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def segment_classical(bgr, config):
    """Return a full-resolution binary uint8 mask {0,255} from pixels only."""
    if bgr is None or bgr.ndim != 3:
        raise ValueError("Expected a readable BGR image")
    original_size = (bgr.shape[1], bgr.shape[0])
    image, _ = resize_for_inference(bgr, int(config.get("max_side", 384)))
    method = config["method"]
    if method == "hsv":
        seed = hsv_seed(image, config)
    elif method == "lab_center":
        seed = lab_center_seed(image, config)
    elif method == "grabcut_rect":
        seed = grabcut_rect(image, config)
    elif method == "hybrid_hsv":
        seed = refine_with_grabcut(image, hsv_seed(image, config), config)
    elif method == "hybrid_lab":
        seed = refine_with_grabcut(image, lab_center_seed(image, config), config)
    else:
        raise ValueError(f"Unknown classical method: {method}")
    seed = morphology(seed, config.get("open_kernel", 3), config.get("close_kernel", 5))
    if config.get("component_filter", True):
        seed = select_components(
            seed,
            min_area_frac=config.get("min_area_frac", 0.001),
            max_area_frac=config.get("max_area_frac", 0.65),
            max_components=config.get("max_components", 4),
            center_weight=config.get("center_weight", 1.0),
        )
    if seed.shape[:2] != bgr.shape[:2]:
        seed = cv2.resize(seed, original_size, interpolation=cv2.INTER_NEAREST)
    return ((seed > 0).astype(np.uint8) * 255)


def metric_counts(prediction, reference):
    pred, ref = prediction > 0, reference > 0
    if pred.shape != ref.shape:
        raise ValueError(f"Mask shape mismatch: {pred.shape} vs {ref.shape}")
    tp = int(np.logical_and(pred, ref).sum())
    fp = int(np.logical_and(pred, np.logical_not(ref)).sum())
    fn = int(np.logical_and(np.logical_not(pred), ref).sum())
    tn = int(np.logical_and(np.logical_not(pred), np.logical_not(ref)).sum())
    return tp, fp, fn, tn


def metrics_from_counts(tp, fp, fn, tn):
    """Empty policy: both-empty scores 1; all other zero-division cases score 0."""
    union = tp + fp + fn
    if union == 0:
        return {"iou": 1.0, "dice": 1.0, "precision": 1.0, "recall": 1.0}
    return {
        "iou": tp / union,
        "dice": (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
    }


def segmentation_metrics(prediction, reference):
    return metrics_from_counts(*metric_counts(prediction, reference))


def overlay_mask(bgr, mask, color=(40, 40, 240), alpha=0.38):
    output = bgr.copy()
    selected = mask > 0
    if selected.any():
        layer = np.zeros_like(output)
        layer[selected] = color
        output[selected] = cv2.addWeighted(output, 1 - alpha, layer, alpha, 0)[selected]
        contours, _ = cv2.findContours((selected.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, (0, 0, 0), 3)
        cv2.drawContours(output, contours, -1, color, 1)
    return output


def comparison_panel(bgr, prediction, reference, image_id, metrics):
    pred_overlay = overlay_mask(bgr, prediction, (30, 30, 240))
    ref_overlay = overlay_mask(bgr, reference, (30, 200, 30))
    error = np.zeros_like(bgr)
    pred, ref = prediction > 0, reference > 0
    error[np.logical_and(pred, ref)] = (60, 200, 60)   # TP green
    error[np.logical_and(pred, ~ref)] = (40, 40, 240)  # FP red
    error[np.logical_and(~pred, ref)] = (240, 160, 30) # FN blue
    canvas = np.concatenate([pred_overlay, ref_overlay, error], axis=1)
    label = f"{image_id} | classical / reference / error | IoU {metrics['iou']:.3f} Dice {metrics['dice']:.3f}"
    cv2.putText(canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas
