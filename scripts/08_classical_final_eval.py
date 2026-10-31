"""Freeze the DEV-selected classical method, then evaluate EVAL ONCE.

The freeze record is written BEFORE eval reference pixels are read. Existing
final results are reused, not silently recomputed. Never tune on these results.

References are human-reviewed SAM-derived silver masks, not independent gold.
Developed with Arena.ai assistance; real execution and disclosure required.
"""

from pathlib import Path
import argparse
from datetime import datetime, timezone
import csv
import json
import sys

import cv2
import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from classical_utils import (file_sha256, segment_classical, metric_counts,
                             metrics_from_counts, comparison_panel)


def now():
    return datetime.now(timezone.utc).isoformat()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError("No rows to write")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(project_root):
    project = Path(project_root)
    metadata, results = project / "metadata", project / "results"
    results.mkdir(exist_ok=True)
    summary_path = results / "classical_eval_summary.json"
    freeze_path = metadata / "classical_eval_freeze.json"
    split_path = metadata / "split.csv"
    selected_path = project / "experiments" / "classical_dev_selected.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        print("Final evaluation already exists and is reused. No re-evaluation or tuning performed.")
        print(json.dumps(summary["eval_macro"], indent=2))
        return summary
    if freeze_path.exists():
        freeze = json.loads(freeze_path.read_text())
        selected = json.loads(selected_path.read_text())
        if file_sha256(selected_path) != freeze["selected_dev_file_sha256"]:
            raise RuntimeError("DEV selection changed after eval freeze. Do not overwrite or retune the final result.")
    else:
        selected = json.loads(selected_path.read_text())
        split_freeze = json.loads((metadata / "split_freeze.json").read_text())
        if file_sha256(split_path) != split_freeze["file_sha256"]["split.csv"]:
            raise RuntimeError("Frozen split changed.")
        if selected["split_sha256"] != file_sha256(split_path):
            raise RuntimeError("Selected config belongs to another split.")
        for name, expected in selected["code_sha256"].items():
            if file_sha256(SCRIPT_DIR / name) != expected:
                raise RuntimeError(f"Classical code changed after DEV selection: {name}. Rerun dev before final eval.")
        status_rows = read_csv(project / "references" / "review_status.csv")
        incomplete = [r["image_id"] for r in status_rows if not r["status"].startswith("reviewed")]
        if len(status_rows) != 110 or incomplete:
            raise RuntimeError(f"All 110 references must be reviewed before final eval. Incomplete: {incomplete[:12]}")
        split_rows = read_csv(split_path)
        reference_hashes = {}
        for row in split_rows:
            path = project / "references" / "masks" / f"{row['image_id']}.png"
            if not path.exists():
                raise RuntimeError(f"Missing reference mask: {row['image_id']}")
            reference_hashes[row["image_id"]] = file_sha256(path)
        # Commit algorithm, code and all reference identities before reading eval pixels.
        freeze = {
            "status": "FROZEN_BEFORE_EVAL_REFERENCE_PIXELS_READ", "frozen_at": now(),
            "selected_config_id": selected["selected_config_id"], "config": selected["config"],
            "selected_dev_file_sha256": file_sha256(selected_path),
            "split_sha256": file_sha256(split_path),
            "reference_policy_sha256": file_sha256(project / "references" / "reference_policy.json"),
            "reference_mask_sha256": reference_hashes,
            "code_sha256": {name: file_sha256(SCRIPT_DIR / name) for name in
                            ("classical_utils.py", "07_classical_dev_experiments.py", "08_classical_final_eval.py")},
            "eval_ids": [r["image_id"] for r in split_rows if r["split"] == "eval"],
            "selection_rule": selected["selection_rule"],
            "prohibition": "Do not change parameters, prompts, split or references in response to eval results. Report post-eval changes separately as exploratory.",
        }
        freeze_path.write_text(json.dumps(freeze, indent=2) + "\n")
        print("Classical algorithm frozen. Now running the 22-image held-out eval.")
    for name, expected in freeze["code_sha256"].items():
        if file_sha256(SCRIPT_DIR / name) != expected:
            raise RuntimeError(f"Code differs from eval freeze: {name}")
    split_rows = read_csv(split_path)
    if file_sha256(split_path) != freeze["split_sha256"]:
        raise RuntimeError("Split changed after classical freeze.")
    eval_rows = [r for r in split_rows if r["split"] == "eval"]
    if len(eval_rows) != 22 or [r["image_id"] for r in eval_rows] != freeze["eval_ids"]:
        raise RuntimeError("Eval IDs differ from freeze.")
    mask_dir = project / "predictions" / "classical" / "eval"
    overlay_dir = project / "predictions" / "classical" / "eval_overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    raw = project / "data" / "raw"
    rows = []
    cv2.setRNGSeed(42)
    for index, row in enumerate(eval_rows, 1):
        image_id = row["image_id"]
        print(f"[{index:02}/22] {image_id}", flush=True)
        image_path = raw / row["relative_path"]
        if file_sha256(image_path) != row["sha256"]:
            raise RuntimeError(f"Raw image changed: {image_id}")
        image = cv2.imread(str(image_path))
        reference_path = project / "references" / "masks" / f"{image_id}.png"
        if file_sha256(reference_path) != freeze["reference_mask_sha256"][image_id]:
            raise RuntimeError(f"Reference changed after freeze: {image_id}")
        reference = np.array(Image.open(reference_path).convert("L"))
        prediction = segment_classical(image, freeze["config"])
        counts = metric_counts(prediction, reference)
        metrics = metrics_from_counts(*counts)
        result = {"image_id": image_id, "scene_group_id": row["scene_group_id"], "split": "eval", **metrics}
        result.update(dict(zip(("tp", "fp", "fn", "tn"), counts)))
        result.update({"reference_area": int((reference > 0).sum()), "prediction_area": int((prediction > 0).sum()),
                       "predicted_empty": int(not (prediction > 0).any()),
                       "reference_empty": int(not (reference > 0).any())})
        rows.append(result)
        Image.fromarray(prediction, mode="L").save(mask_dir / f"{image_id}.png")
        panel = comparison_panel(image, prediction, reference, image_id, metrics)
        cv2.imwrite(str(overlay_dir / f"{image_id}.jpg"), panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
    totals = [sum(int(row[key]) for row in rows) for key in ("tp", "fp", "fn", "tn")]
    macro = {f"mean_{key}": float(np.mean([float(row[key]) for row in rows]))
             for key in ("iou", "dice", "precision", "recall")}
    macro["median_iou"] = float(np.median([float(row["iou"]) for row in rows]))
    micro = metrics_from_counts(*totals)
    write_csv(results / "classical_eval_per_image.csv", rows)
    worst = sorted(rows, key=lambda r: (float(r["iou"]), r["image_id"]))
    write_csv(results / "classical_eval_worst_cases.csv", worst[:10])
    selected = json.loads(selected_path.read_text())
    summary = {
        "status": "FINAL_HELD_OUT_RESULT", "evaluated_at": now(), "eval_images": len(rows),
        "selected_config_id": freeze["selected_config_id"], "config": freeze["config"],
        "dev_metrics_at_selection": selected["dev_metrics"],
        "eval_macro": macro, "eval_micro": micro,
        "total_counts": dict(zip(("tp", "fp", "fn", "tn"), totals)),
        "predicted_empty": sum(int(r["predicted_empty"]) for r in rows),
        "reference_empty": sum(int(r["reference_empty"]) for r in rows),
        "empty_metric_policy": "Both-empty=1; all other zero-division cases=0.",
        "reference_type": "Human-reviewed SAM-derived silver masks; not independent human gold.",
        "split_limitations": "Conservative G11 kitchen family (58 images, including multi-egg scenes) is dev-only; held-out multi-egg and scene diversity are limited. Visual families do not prove physical-egg/session independence.",
        "freeze_sha256": file_sha256(freeze_path),
        "ai_assistance": "Code developed with Arena.ai; results generated by the user's actual run. Disclose assistance.",
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("\nFINAL EVAL COMPLETE - DO NOT TUNE ON THESE RESULTS")
    print(json.dumps(macro, indent=2))
    print("Micro:", json.dumps(micro))
    print("Worst eval IDs for qualitative analysis:", ", ".join(r["image_id"] for r in worst[:10]))
    print("Results saved under results/. Next: full-video tracking, then report packaging.")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default="/content/drive/MyDrive/Egg_CV_Task")
    args = parser.parse_args()
    main(args.project_root)
