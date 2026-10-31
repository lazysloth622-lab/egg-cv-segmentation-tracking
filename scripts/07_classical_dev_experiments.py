"""Run classical segmentation experiments on DEV ONLY and select a config.

This script never reads eval reference masks or eval metrics. It tries a fixed
classical grid, logs every trial, selects by macro DEV IoU/Dice, and saves DEV
predictions/failure cases. Run 08 only when development is finished.

No SAM prompt, reference box, neural network or training is used in inference.
Developed with Arena.ai assistance; inspect, execute and disclose its use.
"""

from pathlib import Path
import argparse
from datetime import datetime, timezone
import csv
import json
import platform
import shutil
import sys
import time

import cv2
import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from classical_utils import (file_sha256, segment_classical, metric_counts,
                             metrics_from_counts, comparison_panel)

BASE = {
    "method": "hsv", "max_side": 384, "sat_max": 80, "val_min": 70,
    "warm": False, "warm_hue_min": 3, "warm_hue_max": 35, "warm_sat_max": 180,
    "center_patch": 0.16, "lab_threshold": 1.2, "lab_scales": [45.0, 22.0, 22.0],
    "rect_scale": 0.70, "grabcut_iters": 4, "sure_fg_kernel": 9, "sure_bg_kernel": 15,
    "open_kernel": 3, "close_kernel": 5, "component_filter": True,
    "min_area_frac": 0.001, "max_area_frac": 0.65, "max_components": 4,
    "center_weight": 1.0,
}


def trial_configs():
    variations = [
        ("C01_hsv_pale", {"method": "hsv", "sat_max": 55, "val_min": 90}),
        ("C02_hsv_medium", {"method": "hsv", "sat_max": 85, "val_min": 65}),
        ("C03_hsv_broad", {"method": "hsv", "sat_max": 120, "val_min": 45, "min_area_frac": 0.0015}),
        ("C04_hsv_warm", {"method": "hsv", "sat_max": 90, "val_min": 55, "warm": True}),
        ("C05_lab_tight", {"method": "lab_center", "lab_threshold": 0.9}),
        ("C06_lab_medium", {"method": "lab_center", "lab_threshold": 1.3}),
        ("C07_lab_broad", {"method": "lab_center", "lab_threshold": 1.8, "min_area_frac": 0.0015}),
        ("C08_grabcut_center", {"method": "grabcut_rect", "rect_scale": 0.50, "max_components": 2}),
        ("C09_grabcut_wide", {"method": "grabcut_rect", "rect_scale": 0.72}),
        ("C10_grabcut_full", {"method": "grabcut_rect", "rect_scale": 0.86, "max_components": 6}),
        ("C11_hybrid_hsv", {"method": "hybrid_hsv", "sat_max": 85, "val_min": 65}),
        ("C12_hybrid_lab", {"method": "hybrid_lab", "lab_threshold": 1.3}),
        ("C13_hybrid_warm", {"method": "hybrid_hsv", "sat_max": 90, "val_min": 55, "warm": True}),
        ("C14_hsv_single", {"method": "hsv", "sat_max": 85, "val_min": 65, "max_components": 1, "center_weight": 2.0}),
        ("C15_lab_single", {"method": "lab_center", "lab_threshold": 1.3, "max_components": 1, "center_weight": 2.0}),
        ("C16_grabcut_slow", {"method": "grabcut_rect", "rect_scale": 0.72, "grabcut_iters": 7, "close_kernel": 7}),
    ]
    output = []
    for name, update in variations:
        config = BASE.copy()
        config.update(update)
        output.append((name, config))
    return output


def now():
    return datetime.now(timezone.utc).isoformat()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(config_id, config, rows, elapsed):
    totals = [sum(int(row[key]) for row in rows) for key in ("tp", "fp", "fn", "tn")]
    micro = metrics_from_counts(*totals)
    result = {
        "config_id": config_id, "method": config["method"], "images": len(rows),
        "mean_iou": float(np.mean([float(r["iou"]) for r in rows])),
        "median_iou": float(np.median([float(r["iou"]) for r in rows])),
        "mean_dice": float(np.mean([float(r["dice"]) for r in rows])),
        "mean_precision": float(np.mean([float(r["precision"]) for r in rows])),
        "mean_recall": float(np.mean([float(r["recall"]) for r in rows])),
        "micro_iou": micro["iou"], "micro_dice": micro["dice"],
        "micro_precision": micro["precision"], "micro_recall": micro["recall"],
        "empty_predictions": sum(int(r["predicted_empty"]) for r in rows),
        "errors": sum(r["status"] != "OK" for r in rows),
        "elapsed_seconds": elapsed, "config_json": json.dumps(config, sort_keys=True),
    }
    return result


def archive_previous(project):
    experiments = project / "experiments"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = experiments / f"archive_{stamp}"
    archive.mkdir(parents=True, exist_ok=False)
    for name in ("classical_dev_per_image.csv", "classical_dev_summary.csv",
                 "classical_dev_selected.json", "classical_dev_run.json",
                 "classical_dev_worst_cases.csv"):
        path = experiments / name
        if path.exists():
            shutil.move(str(path), str(archive / name))
    for name in ("dev", "dev_overlays"):
        path = project / "predictions" / "classical" / name
        if path.exists():
            shutil.move(str(path), str(archive / name))
    print("Previous DEV experiment archived to", archive)


def main(project_root, redo_dev=False):
    project = Path(project_root)
    metadata, experiments = project / "metadata", project / "experiments"
    experiments.mkdir(exist_ok=True)
    if (metadata / "classical_eval_freeze.json").exists():
        raise RuntimeError("Final classical evaluation is already frozen. Do not retune on eval; preserve the reported result.")
    selected_path = experiments / "classical_dev_selected.json"
    if selected_path.exists():
        if not redo_dev:
            raise RuntimeError("DEV selection already exists. Run 08 for final eval, or use --redo-dev BEFORE eval to archive and improve on dev.")
        archive_previous(project)
    split_path = metadata / "split.csv"
    split_freeze = json.loads((metadata / "split_freeze.json").read_text())
    if file_sha256(split_path) != split_freeze["file_sha256"]["split.csv"]:
        raise RuntimeError("Frozen split changed.")
    rows = [r for r in read_csv(split_path) if r["split"] == "dev"]
    if len(rows) != 88:
        raise RuntimeError("Expected 88 DEV images.")
    reference_policy = project / "references" / "reference_policy.json"
    if not reference_policy.exists():
        raise RuntimeError("Run 06 and complete reference review first.")
    missing = [r["image_id"] for r in rows if not (project / "references" / "masks" / f"{r['image_id']}.png").exists()]
    if missing:
        raise RuntimeError(f"Missing reviewed DEV masks ({len(missing)}): {missing[:12]}")
    status = {r["image_id"]: r for r in read_csv(project / "references" / "review_status.csv")}
    not_reviewed = [r["image_id"] for r in rows if not status.get(r["image_id"], {}).get("status", "").startswith("reviewed")]
    if not_reviewed:
        raise RuntimeError(f"Unreviewed DEV references: {not_reviewed[:12]}")
    raw = project / "data" / "raw"
    per_image, summaries, failures = [], [], []
    cv2.setRNGSeed(42)
    start_all = time.time()
    print(f"Running {len(trial_configs())} fixed classical configs on {len(rows)} DEV images. EVAL IS NOT READ.")
    for trial_index, (config_id, config) in enumerate(trial_configs(), 1):
        start = time.time()
        trial_rows = []
        print(f"[{trial_index:02}/{len(trial_configs())}] {config_id} ...", flush=True)
        for row in rows:
            image_id = row["image_id"]
            result = {"config_id": config_id, "image_id": image_id, "split": "dev",
                      "scene_group_id": row["scene_group_id"], "status": "OK"}
            try:
                image_path = raw / row["relative_path"]
                if file_sha256(image_path) != row["sha256"]:
                    raise RuntimeError("raw hash mismatch")
                image = cv2.imread(str(image_path))
                if image is None:
                    raise RuntimeError("unreadable image")
                reference = np.array(Image.open(project / "references" / "masks" / f"{image_id}.png").convert("L"))
                prediction = segment_classical(image, config)
                counts = metric_counts(prediction, reference)
                metrics = metrics_from_counts(*counts)
                result.update(metrics)
                result.update(dict(zip(("tp", "fp", "fn", "tn"), counts)))
                result.update({"reference_area": int((reference > 0).sum()),
                               "prediction_area": int((prediction > 0).sum()),
                               "predicted_empty": int(not (prediction > 0).any()),
                               "reference_empty": int(not (reference > 0).any()), "error": ""})
            except Exception as exc:
                # Failed trials remain visible and cannot win selection.
                result.update({"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0,
                               "tp": 0, "fp": 0, "fn": 0, "tn": 0, "reference_area": 0,
                               "prediction_area": 0, "predicted_empty": 1, "reference_empty": 0,
                               "status": "ERROR", "error": str(exc)})
                failures.append({"config_id": config_id, "image_id": image_id, "error": str(exc)})
            trial_rows.append(result)
            per_image.append(result)
        summary = summarize(config_id, config, trial_rows, time.time() - start)
        summaries.append(summary)
        print(f"    mean IoU {summary['mean_iou']:.4f} | Dice {summary['mean_dice']:.4f} | errors {summary['errors']}", flush=True)
    valid = [s for s in summaries if s["errors"] == 0]
    if not valid:
        raise RuntimeError("Every configuration had an execution error. Inspect the experiment log; no config selected.")
    best = max(valid, key=lambda s: (s["mean_iou"], s["mean_dice"], s["micro_iou"]))
    best_config = json.loads(best["config_json"])
    write_csv(experiments / "classical_dev_summary.csv", summaries)
    write_csv(experiments / "classical_dev_per_image.csv", per_image)
    best_rows = [r for r in per_image if r["config_id"] == best["config_id"]]
    worst = sorted(best_rows, key=lambda r: (float(r["iou"]), r["image_id"]))[:20]
    write_csv(experiments / "classical_dev_worst_cases.csv", worst)

    mask_dir = project / "predictions" / "classical" / "dev"
    overlay_dir = project / "predictions" / "classical" / "dev_overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    print("Saving selected-config DEV predictions and failure overlays...")
    for row in rows:
        image_id = row["image_id"]
        image = cv2.imread(str(raw / row["relative_path"]))
        prediction = segment_classical(image, best_config)
        reference = np.array(Image.open(project / "references" / "masks" / f"{image_id}.png").convert("L"))
        Image.fromarray(prediction, mode="L").save(mask_dir / f"{image_id}.png")
        metric = next(r for r in best_rows if r["image_id"] == image_id)
        panel = comparison_panel(image, prediction, reference, image_id, metric)
        cv2.imwrite(str(overlay_dir / f"{image_id}.jpg"), panel, [cv2.IMWRITE_JPEG_QUALITY, 90])
    selected = {
        "status": "DEV_SELECTED_NOT_YET_EVAL_FROZEN", "selected_at": now(),
        "selection_rule": "Maximum macro DEV mean IoU; ties: mean Dice, then micro IoU. No eval data used.",
        "selected_config_id": best["config_id"], "config": best_config,
        "dev_metrics": best, "split_sha256": file_sha256(split_path),
        "reference_policy_sha256": file_sha256(reference_policy),
        "code_sha256": {name: file_sha256(SCRIPT_DIR / name) for name in ("classical_utils.py", "07_classical_dev_experiments.py")},
        "dev_summary_sha256": file_sha256(experiments / "classical_dev_summary.csv"),
        "dev_per_image_sha256": file_sha256(experiments / "classical_dev_per_image.csv"),
        "empty_metric_policy": "Both-empty=1; all other zero-division cases=0.",
        "ai_assistance": "Classical code and fixed trial grid developed with Arena.ai; actual dev execution selects the config.",
    }
    selected_path.write_text(json.dumps(selected, indent=2) + "\n")
    run = {
        "started_utc": datetime.fromtimestamp(start_all, timezone.utc).isoformat(), "finished_utc": now(),
        "dev_images": len(rows), "eval_images_read": 0, "trial_count": len(summaries),
        "failures": failures, "selected_config_id": best["config_id"],
        "python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__,
        "elapsed_seconds": time.time() - start_all,
    }
    (experiments / "classical_dev_run.json").write_text(json.dumps(run, indent=2) + "\n")
    with (experiments / "classical_experiment_log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(run, sort_keys=True) + "\n")
    print("\nDEV EXPERIMENTS COMPLETE")
    print("Selected:", best["config_id"])
    print(f"DEV mean IoU {best['mean_iou']:.4f} | Dice {best['mean_dice']:.4f} | precision {best['mean_precision']:.4f} | recall {best['mean_recall']:.4f}")
    print("Worst DEV cases:", ", ".join(r["image_id"] for r in worst[:10]))
    print("Next: inspect DEV overlays. To improve before eval, edit the fixed grid and use --redo-dev.")
    print("When development is finished, run 08 ONCE for held-out evaluation. Do not tune on eval.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default="/content/drive/MyDrive/Egg_CV_Task")
    parser.add_argument("--redo-dev", action="store_true")
    args = parser.parse_args()
    main(args.project_root, args.redo_dev)
