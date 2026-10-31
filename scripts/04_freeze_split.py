"""Approve scene families and freeze a reproducible whole-group 88/22 split.

Run after 03_prepare_scene_groups.py. No image or source manifest is changed.
A real person must review and approve the proposed grouping at the prompt.

This code was developed with Arena.ai assistance. Disclose and understand its
use. A no-crossing check concerns the declared visual families; it does not
prove that people, physical eggs or capture sessions are independent.
"""

from pathlib import Path
import argparse
import csv
import hashlib
from io import StringIO
import json
import random
from collections import Counter

EXPECTED_FINGERPRINT = "cdb04474e1663283b6848cb1efea5a248b34f505cd1d98ab85898075ed07521c"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    return fields, rows


def as_csv(rows, fields):
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def fingerprint(rows):
    data = [{key: row[key] for key in ("image_id", "relative_path", "sha256")}
            for row in sorted(rows, key=lambda item: item["image_id"])]
    return digest(json.dumps(data, sort_keys=True, separators=(",", ":")).encode())


def choose_eval_groups(group_sizes, target=22, seed=42):
    """Uniformly sample one group subset among those having exactly target images.

    Dynamic-programming counts avoid repeatedly trying seeds or looking at
    model performance. This function never reads masks or predictions.
    """
    items = sorted(group_sizes.items())
    ways = [[0] * (target + 1) for _ in range(len(items) + 1)]
    ways[0][0] = 1
    for i, (_, weight) in enumerate(items, 1):
        for total in range(target + 1):
            ways[i][total] = ways[i - 1][total]
            if total >= weight:
                ways[i][total] += ways[i - 1][total - weight]
    valid_count = ways[-1][target]
    if not valid_count:
        raise ValueError("No whole-group 88/22 split exists. Stop: do not break a scene family.")
    rng = random.Random(seed)
    selected, remaining = set(), target
    for i in range(len(items), 0, -1):
        group_id, weight = items[i - 1]
        skip = ways[i - 1][remaining]
        take = ways[i - 1][remaining - weight] if remaining >= weight else 0
        if rng.randrange(skip + take) < take:
            selected.add(group_id)
            remaining -= weight
    assert remaining == 0
    return selected, valid_count


def verify_frozen(metadata):
    info = json.loads((metadata / "split_freeze.json").read_text())
    for filename, expected in info["file_sha256"].items():
        path = metadata / filename
        if not path.is_file() or digest(path.read_bytes()) != expected:
            raise RuntimeError(f"Frozen file missing or changed: {filename}. Do not regenerate a new split.")
    _, rows = read_csv(metadata / "split.csv")
    counts = Counter(row["split"] for row in rows)
    if counts != {"dev": 88, "eval": 22} or len({r["image_id"] for r in rows}) != 110:
        raise RuntimeError("Existing frozen split is invalid. Share this error.")
    seen = {}
    for row in rows:
        old = seen.setdefault(row["scene_group_id"], row["split"])
        if old != row["split"]:
            raise RuntimeError("A declared scene group crosses the split.")
    return info


def main(project_root, approval=None):
    project = Path(project_root)
    metadata = project / "metadata"
    freeze_path = metadata / "split_freeze.json"
    final_paths = [metadata / name for name in ("scene_groups.csv", "split.csv", "pilot_images.csv")]
    if freeze_path.exists():
        info = verify_frozen(metadata)
        print("Existing frozen split verified and reused. Nothing was overwritten.")
        print("DEV: 88 | EVAL: 22 | Declared groups crossing: 0")
        print("Eval groups:", ", ".join(info["eval_groups"]))
        return info
    if any(path.exists() for path in final_paths):
        raise RuntimeError("Partial final outputs exist but the freeze record is missing. Share this error; do not delete or resplit.")

    manifest_path = metadata / "manifest.csv"
    draft_path = metadata / "scene_groups_draft.csv"
    policy_path = metadata / "scene_group_policy_draft.json"
    manifest_before = manifest_path.read_bytes()
    _, original = read_csv(manifest_path)
    fields, draft = read_csv(draft_path)
    if len(original) != 110 or len(draft) != 110:
        raise ValueError("Both manifests must contain exactly 110 images.")
    if len({r["image_id"] for r in draft}) != 110 or len({r["image_id"] for r in original}) != 110:
        raise ValueError("Duplicate image IDs detected.")
    dataset_hash = fingerprint(original)
    if dataset_hash != EXPECTED_FINGERPRINT or fingerprint(draft) != dataset_hash:
        raise ValueError("Image IDs, paths or hashes differ from the reviewed dataset. Stop and share this error.")
    policy = json.loads(policy_path.read_text())
    if policy["dataset_fingerprint"] != dataset_hash:
        raise ValueError("Grouping policy belongs to a different dataset.")
    expected = {}
    for group in policy["groups"]:
        for subgroup in group["subgroups"]:
            for image_id in subgroup["image_ids"]:
                if image_id in expected:
                    raise ValueError("Duplicate assignment inside grouping policy.")
                expected[image_id] = group["group_id"]
    if set(expected) != {r["image_id"] for r in draft}:
        raise ValueError("Grouping policy has missing or extra IDs.")
    raw = (project / "data" / "raw").resolve()
    for row in draft:
        if row["scene_group_id"] != expected[row["image_id"]]:
            raise ValueError("Draft CSV and grouping policy disagree. Share any intended grouping correction first.")
        path = (raw / row["relative_path"]).resolve()
        if not path.is_relative_to(raw) or not path.is_file() or digest(path.read_bytes()) != row["sha256"]:
            raise ValueError(f"Raw input missing or changed: {row['image_id']}")

    group_sizes = Counter(r["scene_group_id"] for r in draft)
    if len(group_sizes) != 14:
        raise ValueError("Expected the reviewed 14-family draft. Share intentional changes before freezing.")
    print("Ready to freeze the reviewed 14-family grouping.")
    print("Reminder: kitchen G11 has 58 images, including the multi-egg scenes.")
    print("G11 must stay in dev for an exact 22-image eval; report the limited eval coverage.")
    print("No guarantee of physical-egg/session independence is possible without provenance.")
    if approval is None:
        approval = input("After reviewing the grouped gallery, type APPROVE to freeze (or STOP): ")
    if str(approval).strip().upper() != "APPROVE":
        print("Stopped. No final split was created. Report image-ID grouping corrections before proceeding.")
        return None

    seed, target = 42, 22
    eval_groups, valid_subsets = choose_eval_groups(group_sizes, target, seed)
    draft = sorted(draft, key=lambda row: row["image_id"])
    approved_rows, split_rows = [], []
    for item in draft:
        row = item.copy()
        row["grouping_status"] = "APPROVED_BY_USER_VISUAL_FAMILY_POLICY"
        approved_rows.append(row)
        split_rows.append({
            "image_id": row["image_id"],
            "scene_group_id": row["scene_group_id"],
            "split": "eval" if row["scene_group_id"] in eval_groups else "dev",
            "relative_path": row["relative_path"],
            "source_split": row["source_split"],
            "source_class": row.get("source_class", ""),
            "sha256": row["sha256"],
        })
    counts = Counter(r["split"] for r in split_rows)
    assert counts == {"dev": 88, "eval": 22}
    assert {r["scene_group_id"] for r in split_rows if r["split"] == "dev"}.isdisjoint(eval_groups)

    # Pick three DEV-only pilot cases without changing the frozen split.
    dev_ids = {r["image_id"] for r in split_rows if r["split"] == "dev"}
    chosen = []
    for candidates, reason in [
        (["I001", "I003", "I014", "I101", "I028"], "single egg / basic prompt"),
        (["I007", "I011", "I013", "I012", "I017", "I038"], "clutter or difficult lighting"),
        (["I050", "I052", "I068"], "multiple visible eggs"),
    ]:
        image_id = next((x for x in candidates if x in dev_ids and x not in chosen), None)
        if image_id is None:
            image_id = next(x for x in sorted(dev_ids) if x not in chosen)
        chosen.append(image_id)
    pilot_rows = [{"image_id": x, "split": "dev", "purpose": reason} for x, reason in zip(chosen, [
        "single egg / basic prompt", "clutter or difficult lighting", "multiple visible eggs"
    ])]
    content = {
        "scene_groups.csv": as_csv(approved_rows, fields),
        "split.csv": as_csv(split_rows, list(split_rows[0])),
        "pilot_images.csv": as_csv(pilot_rows, list(pilot_rows[0])),
    }
    info = {
        "status": "FROZEN_BEFORE_SEGMENTATION_EXPERIMENTS",
        "approval": "Explicit interactive APPROVE; visual families, not independently verified capture identities.",
        "seed": seed,
        "algorithm": "Uniform draw from whole-group subsets summing to 22, using DP counts and Python random.Random(42). No outcomes or difficulty scores used.",
        "valid_group_subsets": valid_subsets,
        "dataset_fingerprint": dataset_hash,
        "input_manifest_sha256": digest(manifest_before),
        "draft_groups_sha256": digest(draft_path.read_bytes()),
        "draft_policy_sha256": digest(policy_path.read_bytes()),
        "dev_count": 88, "eval_count": 22,
        "dev_groups": sorted(set(group_sizes) - eval_groups),
        "eval_groups": sorted(eval_groups),
        "group_sizes": dict(sorted(group_sizes.items())),
        "file_sha256": {name: digest(value) for name, value in content.items()},
        "limitations": policy["limitations"][:3],
        "reference_policy": "Label and quality-check all 110 originals using one fixed visible-egg policy. Do not tune classical parameters on eval or pass reference prompts to classical inference.",
        "ai_assistance": "Code and draft grouping developed with Arena.ai; user review and real execution required. Disclose assistance in the final report.",
    }
    for name, value in content.items():
        (metadata / name).write_bytes(value)
    # Commit marker written last: interrupted writes cannot be mistaken for a freeze.
    freeze_path.write_text(json.dumps(info, indent=2) + "\n")
    assert manifest_path.read_bytes() == manifest_before
    verify_frozen(metadata)
    print("\nSPLIT FROZEN")
    print("DEV: 88 | EVAL: 22 | Declared groups crossing: 0")
    print("Eval groups:", ", ".join(sorted(eval_groups)))
    print("DEV-only pilot IDs:", ", ".join(chosen))
    print("Freeze record:", freeze_path)
    print("Save these files with the code. Do not reroll the split after seeing results.")
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default="/content/drive/MyDrive/Egg_CV_Task")
    args = parser.parse_args()
    main(args.project_root)
