"""Create human-reviewed SAM reference masks for all images, with resume support.

Colab GPU workflow:
  %run ".../06_sam_reference_masks.py"

For each image, numbered SAM candidates are displayed. Enter candidate numbers
(e.g. 0 or 0,3), or b for manual boxes. Review the final overlay and press Enter
to accept. Other commands: r=retry, s=skip, q=save-and-quit, empty=no visible egg.

The automatic grid only discovers candidates. Every accepted instance records a
per-egg point+box prompt or human box and a final human review decision.
No training occurs. These are model-derived, human-reviewed SILVER references,
not independent ground truth. Do not pass these prompts to classical inference.

Developed with Arena.ai assistance. Real execution, review and disclosure required.
"""

from pathlib import Path
import argparse
from datetime import datetime, timezone
import csv
import hashlib
import json
import sys

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import display
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from classical_utils import file_sha256, overlay_mask

GENERATOR_CONFIG = {
    "points_per_side": 16,
    "points_per_batch": 64,
    "pred_iou_thresh": 0.72,
    "stability_score_thresh": 0.78,
    "stability_score_offset": 1.0,
    "box_nms_thresh": 0.7,
    "crop_n_layers": 0,
    "min_mask_region_area": 80,
}
MAX_CANDIDATES = 15
COLORS = [(230, 55, 55), (45, 180, 75), (45, 105, 230), (240, 165, 30),
          (170, 70, 200), (20, 180, 180), (240, 80, 160), (120, 120, 120)]


def now():
    return datetime.now(timezone.utc).isoformat()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mask_iou(a, b):
    a, b = a > 0, b > 0
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def candidate_rank(candidate, shape):
    mask = candidate["segmentation"]
    height, width = shape
    area = int(mask.sum())
    fraction = area / (height * width)
    if area < 40 or fraction < 0.0004 or fraction > 0.70:
        return None
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    w, h = x1 - x0 + 1, y1 - y0 + 1
    if w < 5 or h < 5:
        return None
    fill = area / max(1, w * h)
    aspect = min(w, h) / max(w, h)
    distance = np.hypot((xs.mean() - width / 2) / width, (ys.mean() - height / 2) / height)
    # Candidate ranking only prioritizes review order; it is not egg probability.
    score = (0.25 * float(candidate.get("predicted_iou", 0))
             + 0.20 * float(candidate.get("stability_score", 0))
             + 0.20 * min(1.0, fill) + 0.15 * aspect
             + 0.20 * max(0.0, 1 - 2 * distance))
    ranked = dict(candidate)
    ranked["review_rank_score"] = float(score)
    ranked["derived_bbox"] = [int(x0), int(y0), int(x1 + 1), int(y1 + 1)]
    return ranked


def deduplicate(candidates):
    output = []
    for candidate in candidates:
        mask = candidate["segmentation"] > 0
        area = int(mask.sum())
        duplicate = False
        for old in output:
            other = old["segmentation"] > 0
            intersection = np.logical_and(mask, other).sum()
            smaller = min(area, int(other.sum()))
            if smaller and intersection / smaller > 0.96 and mask_iou(mask, other) > 0.80:
                duplicate = True
                break
        if not duplicate:
            output.append(candidate)
    return output


def show_candidates(rgb, candidates, image_id, split):
    plt.close("all")
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = np.asarray(axes).reshape(-1)
    axes[0].imshow(rgb)
    axes[0].set_title(f"{image_id} | {split} | ORIGINAL")
    axes[0].grid(alpha=0.25)
    for index, axis in enumerate(axes[1:]):
        axis.axis("off")
        if index >= len(candidates):
            continue
        candidate = candidates[index]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        overlay = overlay_mask(bgr, candidate["segmentation"], color=(230, 50, 50))
        axis.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        area = 100 * float(candidate["segmentation"].mean())
        axis.set_title(f"[{index}] area {area:.1f}% | SAM {candidate.get('predicted_iou', 0):.2f}")
    fig.suptitle("Select ALL visible egg candidates; avoid background, hands and merged multi-egg masks", fontsize=15)
    fig.tight_layout()
    display(fig)
    plt.close(fig)


def show_final(rgb, instances, image_id):
    canvas = rgb.copy()
    for index, mask in enumerate(instances, 1):
        color = COLORS[(index - 1) % len(COLORS)]
        selected = mask > 0
        layer = np.zeros_like(canvas)
        layer[selected] = color
        canvas[selected] = (0.55 * canvas[selected] + 0.45 * layer[selected]).astype(np.uint8)
        contours, _ = cv2.findContours(selected.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, 2)
        ys, xs = np.where(selected)
        if len(xs):
            cv2.putText(canvas, str(index), (int(xs.mean()), int(ys.mean())),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
            cv2.putText(canvas, str(index), (int(xs.mean()), int(ys.mean())),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)
    plt.close("all")
    fig, axis = plt.subplots(figsize=(7, 7))
    axis.imshow(canvas)
    axis.set_title(f"{image_id}: {len(instances)} instance(s) — review, then accept or retry")
    axis.axis("off")
    display(fig)
    plt.close(fig)


def internal_point(mask):
    distance = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
    return [float(x), float(y)]


def prompt_from_candidate(predictor, candidate):
    seed = candidate["segmentation"] > 0
    box = np.array(candidate["derived_bbox"], dtype=np.float32)
    point = np.array([internal_point(seed)], dtype=np.float32)
    masks, scores, _ = predictor.predict(point_coords=point, point_labels=np.array([1], dtype=np.int32),
                                         box=box, multimask_output=True)
    similarities = [mask_iou(mask, seed) for mask in masks]
    best = max(range(len(masks)), key=lambda i: (similarities[i], float(scores[i])))
    prompt = {
        "source": "human_selected_SAM_grid_candidate",
        "point_coords": point.tolist(), "point_labels": [1], "box_xyxy": box.tolist(),
        "grid_candidate_point_coords": candidate.get("point_coords", []),
        "candidate_predicted_iou": float(candidate.get("predicted_iou", 0)),
        "candidate_stability_score": float(candidate.get("stability_score", 0)),
        "selected_output_index": best,
        "refined_mask_score": float(scores[best]),
        "seed_agreement_iou": float(similarities[best]),
    }
    return masks[best], prompt


def parse_boxes(text, width, height):
    boxes = []
    for part in text.split(";"):
        values = [float(x.strip()) for x in part.split(",")]
        if len(values) != 4:
            raise ValueError("Each box needs x1,y1,x2,y2")
        x1, y1, x2, y2 = values
        x1, x2 = sorted((max(0, min(width - 1, x1)), max(0, min(width - 1, x2))))
        y1, y2 = sorted((max(0, min(height - 1, y1)), max(0, min(height - 1, y2))))
        if x2 - x1 < 3 or y2 - y1 < 3:
            raise ValueError("Box is too small")
        boxes.append([x1, y1, x2, y2])
    if not boxes:
        raise ValueError("No boxes entered")
    return boxes


def prompt_from_box(predictor, box):
    masks, scores, _ = predictor.predict(box=np.array(box, dtype=np.float32), multimask_output=True)
    best = int(np.argmax(scores))
    return masks[best], {
        "source": "human_box_prompt", "point_coords": [], "point_labels": [],
        "box_xyxy": [float(x) for x in box], "selected_output_index": best,
        "refined_mask_score": float(scores[best]),
    }


def compile_metadata(project, split_rows):
    reference = project / "references"
    records, status = [], []
    for row in split_rows:
        path = reference / "annotations" / f"{row['image_id']}.json"
        if path.exists():
            record = json.loads(path.read_text())
            records.append(record)
            status.append({
                "image_id": row["image_id"], "split": row["split"],
                "status": record["status"], "instance_count": len(record["instances"]),
                "union_area": int(record["union_area"]),
                "raw_sha256": row["sha256"], "reviewed_at": record["reviewed_at"],
            })
        else:
            status.append({"image_id": row["image_id"], "split": row["split"], "status": "PENDING",
                           "instance_count": "", "union_area": "", "raw_sha256": row["sha256"], "reviewed_at": ""})
    with (reference / "prompts.jsonl").open("w", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda x: x["image_id"]):
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    write_csv(reference / "review_status.csv", status,
              ["image_id", "split", "status", "instance_count", "union_area", "raw_sha256", "reviewed_at"])
    return records, status


def save_annotation(project, row, rgb, instances, prompts, status="reviewed"):
    image_id = row["image_id"]
    reference = project / "references"
    for folder in ("masks", "instances", "overlays", "annotations"):
        (reference / folder).mkdir(parents=True, exist_ok=True)
    union = np.any(np.stack(instances), axis=0) if instances else np.zeros(rgb.shape[:2], dtype=bool)
    instance_image = np.zeros(rgb.shape[:2], dtype=np.uint16)
    for index, mask in enumerate(instances, 1):
        # Later instances win overlaps; the original masks and prompt record remain available.
        instance_image[mask > 0] = index
    mask_path = reference / "masks" / f"{image_id}.png"
    instance_path = reference / "instances" / f"{image_id}.png"
    Image.fromarray(union.astype(np.uint8) * 255, mode="L").save(mask_path)
    Image.fromarray(instance_image, mode="I;16").save(instance_path)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay = bgr.copy()
    for index, mask in enumerate(instances):
        overlay = overlay_mask(overlay, mask, color=COLORS[index % len(COLORS)])
    cv2.imwrite(str(reference / "overlays" / f"{image_id}.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
    details = []
    for index, (mask, prompt) in enumerate(zip(instances, prompts), 1):
        ys, xs = np.where(mask > 0)
        details.append({
            "instance_index": index, "area": int((mask > 0).sum()),
            "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else [],
            "prompt": prompt,
        })
    record = {
        "image_id": image_id, "split": row["split"], "relative_path": row["relative_path"],
        "raw_sha256": row["sha256"], "status": status, "reviewed_at": now(),
        "label_policy": "visible_egg_pixels_only_v1", "instances": details,
        "union_area": int(union.sum()), "mask_sha256": file_sha256(mask_path),
        "instance_mask_sha256": file_sha256(instance_path),
        "mask_relative_path": mask_path.relative_to(project).as_posix(),
        "instance_mask_relative_path": instance_path.relative_to(project).as_posix(),
    }
    (reference / "annotations" / f"{image_id}.json").write_text(json.dumps(record, indent=2) + "\n")


def main(project_root, only="", redo=False, max_images=None):
    project = Path(project_root)
    metadata = project / "metadata"
    split_path = metadata / "split.csv"
    freeze = json.loads((metadata / "split_freeze.json").read_text())
    if file_sha256(split_path) != freeze["file_sha256"]["split.csv"]:
        raise RuntimeError("Frozen split changed. Stop; do not resplit.")
    environment = json.loads((metadata / "sam_environment.json").read_text())
    if environment["split_sha256"] != file_sha256(split_path):
        raise RuntimeError("SAM environment belongs to another split.")
    split_rows = read_csv(split_path)
    selected = {x.strip().upper() for x in only.split(",") if x.strip()}
    if selected:
        unknown = selected - {r["image_id"] for r in split_rows}
        if unknown:
            raise ValueError(f"Unknown IDs: {sorted(unknown)}")
        split_rows = [r for r in split_rows if r["image_id"] in selected]
    if redo and not selected:
        raise ValueError("For safety, --redo requires --only I001,I002,...")
    if max_images:
        split_rows = split_rows[:max_images]
    reference = project / "references"
    reference.mkdir(exist_ok=True)
    pending = []
    for row in split_rows:
        annotation = reference / "annotations" / f"{row['image_id']}.json"
        if annotation.exists() and not redo:
            continue
        pending.append(row)
    compile_metadata(project, read_csv(split_path))
    print(f"Selected this run: {len(split_rows)} | already reviewed/skipped: {len(split_rows)-len(pending)} | to review: {len(pending)}")
    if not pending:
        print("Nothing pending. Use --only IDs --redo to correct an accepted mask.")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for the full batch. Enable T4, mount Drive and rerun.")
    from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
    checkpoint = project / environment["checkpoint_relative_path"]
    if file_sha256(checkpoint) != environment["checkpoint"]["sha256"]:
        raise RuntimeError("Checkpoint hash mismatch.")
    policy_path = reference / "reference_policy.json"
    policy = {
        "policy_version": "visible_egg_pixels_only_v1", "status": "HUMAN_REVIEWED_SILVER_NOT_INDEPENDENT_GOLD",
        "model_type": "vit_b", "sam_commit": environment["sam_commit"],
        "checkpoint_sha256": environment["checkpoint"]["sha256"], "generator": GENERATOR_CONFIG,
        "split_sha256": file_sha256(split_path),
        "rule": "Label every visible egg's visible pixels; exclude hand/background. Do not complete hidden regions. Record each prompt and review decision. Empty masks are permitted only after explicit EMPTY confirmation.",
        "eval_use": "Eval references may be scored only after the classical algorithm is frozen; never used for prompts, tuning or model selection.",
        "ai_assistance": "Arena.ai assisted code and candidate interface; human selection/review and real SAM execution required.",
    }
    if policy_path.exists():
        old = json.loads(policy_path.read_text())
        for key in ("policy_version", "sam_commit", "checkpoint_sha256", "generator", "split_sha256"):
            if old[key] != policy[key]:
                raise RuntimeError(f"Reference policy changed: {key}. Do not mix mask versions.")
    else:
        policy_path.write_text(json.dumps(policy, indent=2) + "\n")
    print("Loading pretrained SAM ViT-B. No training will occur.")
    model = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
    model.to("cuda")
    model.eval()
    predictor = SamPredictor(model)
    generator = SamAutomaticMaskGenerator(model=model, output_mode="binary_mask", **GENERATOR_CONFIG)
    raw = project / "data" / "raw"
    for position, row in enumerate(pending, 1):
        path = raw / row["relative_path"]
        if file_sha256(path) != row["sha256"]:
            raise RuntimeError(f"Raw image changed: {row['image_id']}")
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise RuntimeError(f"Unreadable image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        print(f"\n[{position}/{len(pending)}] {row['image_id']} | {row['split']} | {row['relative_path']}")
        with torch.inference_mode():
            generated = generator.generate(rgb)
        ranked = [candidate_rank(item, rgb.shape[:2]) for item in generated]
        candidates = [x for x in ranked if x is not None]
        candidates.sort(key=lambda x: x["review_rank_score"], reverse=True)
        candidates = deduplicate(candidates)[:MAX_CANDIDATES]
        while True:
            show_candidates(rgb, candidates, row["image_id"], row["split"])
            print("\n>>> YOUR TURN: type a candidate number (0 is often the first), then Enter. <<<", flush=True)
            command = input("RESPONSE (0 or 0,3 | b=boxes | r=view | s=skip | q=quit | empty=no egg): ").strip().lower()
            if command == "q":
                compile_metadata(project, read_csv(split_path))
                print("Stopped safely. Rerun to continue from the next pending image.")
                return
            if command == "s":
                print("Skipped; remains PENDING.")
                break
            if command == "r":
                continue
            if command == "empty":
                if input("Type EMPTY to confirm there is no visible egg: ").strip().upper() == "EMPTY":
                    save_annotation(project, row, rgb, [], [], status="reviewed_empty")
                    compile_metadata(project, read_csv(split_path))
                    break
                continue
            try:
                with torch.inference_mode():
                    predictor.set_image(rgb)
                    if command == "b":
                        print("Use ORIGINAL panel axes. One box per egg; semicolon between eggs.")
                        boxes = parse_boxes(input("Boxes x1,y1,x2,y2; x1,y1,x2,y2: "), rgb.shape[1], rgb.shape[0])
                        results = [prompt_from_box(predictor, box) for box in boxes]
                    else:
                        indices = [int(x.strip()) for x in command.split(",")]
                        if not indices or len(set(indices)) != len(indices) or any(x < 0 or x >= len(candidates) for x in indices):
                            raise ValueError("Enter unique candidate numbers shown in the panels")
                        results = [prompt_from_candidate(predictor, candidates[x]) for x in indices]
                instances = [(mask > 0).astype(np.uint8) for mask, _ in results]
                prompts = [prompt for _, prompt in results]
                show_final(rgb, instances, row["image_id"])
                print("\n>>> REVIEW FINAL MASK: press Enter to accept, or type r. <<<", flush=True)
                decision = input("FINAL RESPONSE (Enter=accept | r=retry | s=skip | q=quit): ").strip().lower()
                if decision == "q":
                    compile_metadata(project, read_csv(split_path))
                    return
                if decision == "s":
                    break
                if decision == "r":
                    continue
                if decision == "":
                    save_annotation(project, row, rgb, instances, prompts)
                    compile_metadata(project, read_csv(split_path))
                    print(f"Saved {row['image_id']} with {len(instances)} instance(s).")
                    break
            except (ValueError, RuntimeError) as exc:
                print("Input/prompt error:", exc, "- try again.")
    records, status = compile_metadata(project, read_csv(split_path))
    reviewed = sum(x["status"].startswith("reviewed") for x in status)
    print(f"\nReviewed: {reviewed}/110 | Pending: {110-reviewed}")
    print("Reference masks, instance masks, overlays and prompts are saved under references/.")
    if reviewed == 110:
        print("ALL REFERENCES COMPLETE. Next: 07 classical DEV experiments. No eval tuning.")
    else:
        print("Rerun this script to resume; do not run final evaluation until references are complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default="/content/drive/MyDrive/Egg_CV_Task")
    parser.add_argument("--only", default="", help="Comma-separated IDs, e.g. I001,I050")
    parser.add_argument("--redo", action="store_true", help="Redo only the IDs explicitly supplied with --only")
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()
    main(args.project_root, args.only, args.redo, args.max_images)
