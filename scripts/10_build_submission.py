"""Build an honest report, reproducible repository folder and submission ZIP.

Run after 09 with --final. Missing outputs are marked PENDING in draft mode;
metrics are never invented. Public GitHub/Drive links still require your manual
upload and sharing steps. No credentials are requested.

Developed with Arena.ai assistance. Review every claim and disclose AI use.
"""

from pathlib import Path
import argparse
from datetime import datetime, timezone
import base64
import csv
import hashlib
import html
import json
import mimetypes
import shutil
import subprocess
import sys
import zipfile
from urllib.parse import urlparse

import importlib.metadata

SCRIPT_DIR_GLOBAL = Path(__file__).resolve().parent
if str(SCRIPT_DIR_GLOBAL) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR_GLOBAL))
from report_pdf import build_pdf


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def read_csv(path):
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fmt(value):
    return f"{float(value):.4f}" if value is not None else "PENDING"


def md_table(headers, rows):
    def clean(x):
        return str(x).replace("|", "\\|").replace("\n", " ")
    return "| " + " | ".join(map(clean, headers)) + " |\n|" + "|".join(["---"] * len(headers)) + "|\n" + "".join(
        "| " + " | ".join(clean(x) for x in row) + " |\n" for row in rows)


def image_uri(path):
    if not Path(path).exists():
        return ""
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


SCRIPT_FILES = (
    "01_prepare_manifest.py", "02_make_contact_sheets.py", "03_prepare_scene_groups.py",
    "04_freeze_split.py", "05_setup_sam.py", "06_sam_reference_masks.py",
    "06_sam_widget.py", "07_classical_dev_experiments.py", "08_classical_final_eval.py",
    "09_track_full_video.py", "10_build_submission.py", "classical_utils.py", "report_pdf.py",
)


def validate_final_urls(video_url, github_url):
    video = urlparse(video_url)
    github = urlparse(github_url)
    if video.scheme != "https" or video.netloc.lower() not in {"drive.google.com", "docs.google.com"}:
        raise ValueError("--video-link must be the public HTTPS Google Drive URL for the annotated output video.")
    if github.scheme != "https" or github.netloc.lower() not in {"github.com", "www.github.com"} or len(github.path.strip("/").split("/")) < 2:
        raise ValueError("--github-link must be the public HTTPS repository URL, for example https://github.com/user/repo")
    if "PASTE_" in video_url or "PASTE_" in github_url:
        raise ValueError("Replace link placeholders with the real public URLs.")


def inspect_tracking_video(project, video):
    """Select the accepted corrected output and verify its container metadata."""
    if video.get("actual_tracker") != "detection_gated_hsv":
        raise RuntimeError(
            f"Final packaging expects the visually accepted detection_gated_hsv run, not {video.get('actual_tracker')!r}. "
            "Do not submit the rejected template output."
        )
    candidates = [
        project / "results" / "video_tracked_detection_v2.mp4",
        project / "video_tracked_detection_v2.mp4",
        project / "results" / "video_tracked.mp4",
    ]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            "Accepted annotated MP4 not found. Expected results/video_tracked_detection_v2.mp4 "
            "(preferred) or results/video_tracked.mp4 from the corrected detection run."
        )
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open annotated video: {path}")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    expected_frames = int(video.get("frames", video.get("frames_processed", 0)))
    if frames != expected_frames or frames != int(video.get("frames_processed", 0)):
        raise RuntimeError(f"Annotated video has {frames} frames but tracking summary expects {expected_frames}; refusing to mix outputs.")
    if video.get("fps") and abs(fps - float(video["fps"])) > 0.05:
        raise RuntimeError(f"Annotated video FPS {fps} differs from summary FPS {video['fps']}.")
    if video.get("width") and (width != int(video["width"]) or height != int(video["height"])):
        raise RuntimeError("Annotated video dimensions differ from the tracking summary.")
    return path, {
        "file_name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path),
        "frames": frames, "fps": fps, "width": width, "height": height,
        "note": "The public Drive URL must open this same accepted file; verify manually in an incognito window.",
    }


def verify_final_pdf(pdf_path, video_url, github_url):
    """Check link annotations and unresolved placeholders in the actual PDF."""
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "pypdf"])
        from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    uris = []
    for page in reader.pages:
        for reference in page.get("/Annots", []):
            annotation = reference.get_object()
            action = annotation.get("/A")
            if action and action.get("/URI"):
                uris.append(str(action.get("/URI")))
    missing_links = [url for url in (video_url, github_url) if url not in uris]
    if missing_links:
        raise RuntimeError(f"PDF link verification failed for: {missing_links}. Found URI annotations: {uris}")
    unresolved = [token for token in ("PASTE_PUBLIC", "<PUBLIC_DRIVE", "<PUBLIC_GITHUB", "PENDING") if token in text]
    if unresolved:
        raise RuntimeError(f"Final PDF still contains placeholders: {unresolved}")
    if len(reader.pages) < 4:
        raise RuntimeError("Final PDF is unexpectedly short; refusing to package it.")
    return {"pages": len(reader.pages), "uri_annotations": len(uris)}


def copy_tree_contents(source, destination):
    source, destination = Path(source), Path(destination)
    if not source.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def main(project_root, final=False, video_link="", github_link=""):
    project = Path(project_root)
    video_url = video_link.strip() or "PASTE_PUBLIC_DRIVE_VIEWER_LINK_HERE"
    github_url = github_link.strip() or "PASTE_PUBLIC_GITHUB_LINK_HERE"
    if final:
        validate_final_urls(video_url, github_url)
    metadata, report_dir = project / "metadata", project / "report"
    report_dir.mkdir(exist_ok=True)
    split = read_json(metadata / "split_freeze.json")
    sam = read_json(metadata / "sam_environment.json")
    reference_policy = read_json(project / "references" / "reference_policy.json")
    review = read_csv(project / "references" / "review_status.csv")
    selected = read_json(project / "experiments" / "classical_dev_selected.json")
    trials = read_csv(project / "experiments" / "classical_dev_summary.csv")
    evaluation = read_json(project / "results" / "classical_eval_summary.json")
    eval_rows = read_csv(project / "results" / "classical_eval_per_image.csv")
    video = read_json(project / "results" / "video_tracking_summary.json")
    reviewed = sum(r["status"].startswith("reviewed") for r in review)
    missing = []
    video_info = {}
    video_path = None
    if not split: missing.append("frozen split")
    if not sam: missing.append("SAM environment/checkpoint record")
    if not reference_policy: missing.append("reference policy")
    if reviewed != 110: missing.append(f"reviewed references ({reviewed}/110)")
    if not selected: missing.append("classical DEV selection")
    if not trials: missing.append("classical DEV trial log")
    if final and len(trials) != 16: missing.append(f"16 classical DEV trials ({len(trials)}/16)")
    if not evaluation: missing.append("held-out evaluation")
    if final and evaluation and int(evaluation.get("eval_images", 0)) != 22:
        missing.append(f"22 held-out evaluation images ({evaluation.get('eval_images', 0)}/22)")
    if final and len(eval_rows) != 22:
        missing.append(f"22 per-image EVAL rows ({len(eval_rows)}/22)")
    if not video or video.get("status") != "FULL_VIDEO_TRACKED":
        missing.append("full-video tracking")
    elif final:
        try:
            video_path, video_info = inspect_tracking_video(project, video)
        except Exception as exc:
            missing.append(f"accepted corrected annotated video ({exc})")
    if final and missing:
        raise RuntimeError("Cannot finalize; missing: " + ", ".join(missing))
    status = "COMPLETE_LOCAL_RESULTS" if not missing else "DRAFT_INCOMPLETE"

    dev = selected.get("dev_metrics", {}) if selected else {}
    ev = evaluation.get("eval_macro", {}) if evaluation else {}
    micro = evaluation.get("eval_micro", {}) if evaluation else {}
    config = selected.get("config", {}) if selected else {}
    if video and video.get("actual_tracker") == "detection_gated_hsv":
        tracking_note = ("Detection-gated tracking uses HSV/contour candidates, shape/size gates, one-to-one temporal association and green-card suppression. "
                         "not_detected and out_of_scene rows have blank coordinates; boxes are hidden rather than drifted or interpolated.")
    else:
        tracking_note = "User-provided boxes at candidate segment starts preserve labels; lost tracks are marked rather than interpolated."
    trial_table = []
    for row in sorted(trials, key=lambda r: float(r["mean_iou"]), reverse=True):
        trial_table.append([row["config_id"], row["method"], fmt(row["mean_iou"]), fmt(row["mean_dice"]),
                            fmt(row["mean_precision"]), fmt(row["mean_recall"]), row["errors"]])
    worst = sorted(eval_rows, key=lambda r: float(r["iou"]))[:10]
    commands = f"""from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/Egg_CV_Task
%run scripts/01_prepare_manifest.py
%run scripts/02_make_contact_sheets.py
%run scripts/03_prepare_scene_groups.py
%run scripts/04_freeze_split.py
%run scripts/05_setup_sam.py
%run scripts/06_sam_widget.py
%run scripts/07_classical_dev_experiments.py
%run scripts/08_classical_final_eval.py
%run scripts/09_track_full_video.py --tracker detection
%run scripts/10_build_submission.py --final --video-link '{video_url}' --github-link '{github_url}'"""
    # Recommended: Colab. CPU box-only SAM fallback is documented in REPRODUCE.md.
    md = f"""# Egg Segmentation and Tracking — Execution Report

**Status:** {status}  
**Generated UTC:** {now()}  
**AI assistance:** Planning, draft visual grouping, code and documentation were assisted by Arena.ai. The user executed the pipeline and was responsible for prompt selection, mask review and final verification. No result in this report is intended to represent unexecuted work.

## 1. Task and dataset

- 110 original 640×640 JPG images from the supplied Roboflow v1 classification dataset; original train/valid/test labels were retained only as source metadata.
- The task uses pretrained SAM for reference-mask creation, a separate classical segmentation method, group-aware development/evaluation, and individual egg tracking on the full supplied video.
- The source dataset has no independent pixel-level gold masks. References here are **human-reviewed SAM-derived silver masks**, so classical scores measure agreement with those references and are not absolute accuracy against independent human ground truth.

## 2. Leakage-reduction split

- Visual scene families were grouped before segmentation experiments; 88 development and 22 held-out evaluation images were frozen with seed 42.
- Declared scene groups crossing the split: **0**.
- Evaluation groups: **{', '.join(split.get('eval_groups', [])) if split else 'PENDING'}**.
- Conservative kitchen family G11 has 58 images, including the clearly visible multi-egg scenes, and remains development-only. Therefore held-out multi-egg and scene diversity are limited. This is reported rather than breaking related scenes to force coverage.
- Visual families reduce obvious setup leakage but cannot prove unique people, physical eggs or capture sessions without provenance.

## 3. SAM reference creation

- Model: **pretrained SAM ViT-B**; no SAM training or fine-tuning was performed.
- Official repository commit: `{sam.get('sam_commit', 'PENDING') if sam else 'PENDING'}`.
- Checkpoint SHA-256: `{sam.get('checkpoint', {}).get('sha256', 'PENDING') if sam else 'PENDING'}`.
- Reviewed references: **{reviewed}/110**.
- Automatic SAM grid candidates were used for prompt discovery. Accepted instances record per-egg point+box or human-box prompts and a final human review decision.
- Label policy: visible egg pixels only; hands/background excluded; hidden egg regions were not completed. Binary union and per-instance PNGs, prompts, overlays and review status are saved.

## 4. Classical method and DEV-only experiments

Classical inference receives only the original image and a frozen configuration. It does **not** receive SAM prompts, reference boxes or reference masks. Operations include HSV/Lab appearance seeds, morphology, connected-component filtering and/or GrabCut depending on the selected configuration.

- Fixed configurations tried on development images: **{len(trials)}**.
- Selection rule: maximum macro development mean IoU; ties broken by mean Dice then micro IoU.
- Selected configuration: **{selected.get('selected_config_id', 'PENDING') if selected else 'PENDING'}**.
- Evaluation references and metrics were not used for selection. The final algorithm was frozen before evaluation reference pixels were read.

```json
{json.dumps(config, indent=2)}
```

### Development trial summary

{md_table(['Config','Method','IoU','Dice','Precision','Recall','Errors'], trial_table) if trial_table else 'PENDING'}

## 5. Results

{md_table(['Split / aggregation','IoU','Dice','Precision','Recall'], [
 ['DEV macro', fmt(dev.get('mean_iou')), fmt(dev.get('mean_dice')), fmt(dev.get('mean_precision')), fmt(dev.get('mean_recall'))],
 ['EVAL macro', fmt(ev.get('mean_iou')), fmt(ev.get('mean_dice')), fmt(ev.get('mean_precision')), fmt(ev.get('mean_recall'))],
 ['EVAL micro', fmt(micro.get('iou')), fmt(micro.get('dice')), fmt(micro.get('precision')), fmt(micro.get('recall'))],
])}

Metric definitions: **IoU = TP/(TP+FP+FN)**; **Dice = 2TP/(2TP+FP+FN)**; **precision = TP/(TP+FP)**; **recall = TP/(TP+FN)**. Macro averages per-image metrics with equal image weight; micro pools pixel counts before calculating the metric. These are overlap measures, not classification accuracy.

Empty-mask metric policy: both prediction and reference empty scores 1; other zero-division cases score 0. Per-image counts, TP/FP/FN/TN and masks are retained. EVAL median IoU: **{fmt(ev.get('median_iou'))}**.

### Lowest-IoU held-out cases for qualitative review

{md_table(['Image','Group','IoU','Dice','Prediction area','Reference area'], [[r['image_id'],r['scene_group_id'],fmt(r['iou']),fmt(r['dice']),r['prediction_area'],r['reference_area']] for r in worst]) if worst else 'PENDING'}

Inspect the saved three-panel overlays (classical/reference/error). Report observable failure modes such as background-color confusion, occlusion, highlights, multiple-object merging or missed objects only when actually visible. Do not infer causes solely from a metric.

## 6. Full-video individual tracking

- Status: **{video.get('status', 'PENDING') if video else 'PENDING'}**.
- Frames processed: **{video.get('frames_processed', 'PENDING') if video else 'PENDING'} / {video.get('frames', 'PENDING') if video else 'PENDING'}**.
- Tracker: **{video.get('actual_tracker', 'PENDING') if video else 'PENDING'}**.
- Persistent object labels: **{', '.join(video.get('object_ids', [])) if video else 'PENDING'}**.
- Candidate segments: **{video.get('candidate_segments', 'PENDING') if video else 'PENDING'}**; not-detected object-frames: **{video.get('lost_object_frames', 'PENDING') if video else 'PENDING'}**; out-of-scene object-frames: **{video.get('out_of_scene_object_frames', 'not recorded') if video else 'PENDING'}**.
- {tracking_note} OpenCV output has no audio.
- Selected annotated file: **{video_info.get('file_name', 'PENDING')}**; verified container frames: **{video_info.get('frames', 'PENDING')}**; FPS: **{video_info.get('fps', 'PENDING')}**; resolution: **{video_info.get('width', 'PENDING')}×{video_info.get('height', 'PENDING')}**.
- Annotated-file SHA-256: `{video_info.get('sha256', 'PENDING')}`. The MP4 stays on Drive, not in GitHub; this hash identifies the file that must be linked.
- Cut detection and tracking can fail under occlusion, abrupt motion or ambiguous identity; inspect the complete rendered video and disclose observed issues. Execution counts are not proof of visual correctness or tracking accuracy.

**Public annotated-video link:** `PASTE_PUBLIC_DRIVE_VIEWER_LINK_HERE`

## 7. Reproducibility

Use a Colab GPU runtime for SAM. Mount Drive, place scripts in `scripts/`, and run in order with the project root `/content/drive/MyDrive/Egg_CV_Task` (the delivered scripts default to that root):

```python
{commands}
```

The video must be placed at `data/video/input.mp4`. Raw data, the SAM checkpoint and large video files are excluded from GitHub; hashes and relative paths are recorded. Exact package/GPU details are in `metadata/sam_environment.json`.

## 8. Limitations and integrity

1. SAM-derived references are not independent gold and may contain review errors.
2. The broad visual grouping is conservative and does not guarantee capture/session independence.
3. Held-out scene and multi-egg diversity are limited by whole-group splitting.
4. Classical development selection can overfit the 88-image development set; held-out results must not be used for further tuning.
5. Video boxes are manual at segment starts and tracking is not a learned raw/boiled classifier.
6. AI assistance is disclosed above; generated code and results require human verification.

## 9. Submission links

- Public GitHub repository: `PASTE_PUBLIC_GITHUB_LINK_HERE`
- Public Drive annotated video: `PASTE_PUBLIC_DRIVE_VIEWER_LINK_HERE`
- Original assignment: CV Intern Task Wellfound.pdf

**Local package status:** {status}. Public accessibility and final submission remain manual checks.
"""
    md = md.replace("PASTE_PUBLIC_DRIVE_VIEWER_LINK_HERE", video_url)
    md = md.replace("PASTE_PUBLIC_GITHUB_LINK_HERE", github_url)
    (report_dir / "REPORT.md").write_text(md)

    css = """body{font:16px/1.6 system-ui,sans-serif;max-width:1100px;margin:30px auto;padding:0 22px;color:#17253b}h1,h2{color:#12345a}table{border-collapse:collapse;width:100%;margin:15px 0}th,td{border:1px solid #d5dfeb;padding:8px;text-align:left}th{background:#edf3fa}.status{padding:12px;background:#fff3cd;border-left:5px solid #c58b12}pre{white-space:pre-wrap;background:#f3f6fa;padding:14px;border-radius:8px}img{max-width:100%;border:1px solid #ccd6e4;margin:10px 0}.note{color:#53647a}"""
    metric_rows = [
        ("DEV macro", dev.get("mean_iou"), dev.get("mean_dice"), dev.get("mean_precision"), dev.get("mean_recall")),
        ("EVAL macro", ev.get("mean_iou"), ev.get("mean_dice"), ev.get("mean_precision"), ev.get("mean_recall")),
        ("EVAL micro", micro.get("iou"), micro.get("dice"), micro.get("precision"), micro.get("recall")),
    ]
    metric_html = "".join(f"<tr><td>{html.escape(name)}</td>" + "".join(f"<td>{fmt(v)}</td>" for v in values) + "</tr>" for name, *values in metric_rows)
    trial_html = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>" for row in trial_table)
    worst_html = "".join(f"<tr><td>{r['image_id']}</td><td>{r['scene_group_id']}</td><td>{fmt(r['iou'])}</td><td>{fmt(r['dice'])}</td></tr>" for r in worst)
    images_html = ""
    for row in worst[:3]:
        uri = image_uri(project / "predictions" / "classical" / "eval_overlays" / f"{row['image_id']}.jpg")
        if uri:
            images_html += f"<h3>{row['image_id']} — IoU {fmt(row['iou'])}</h3><img src='{uri}' alt='Evaluation overlay {row['image_id']}'>"
    page = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>Egg CV Report</title><style>{css}</style></head><body>
<h1>Egg Segmentation and Tracking</h1><p class='status'><b>{status}</b> — generated {now()}.</p>
<p><b>AI-use disclosure:</b> Arena.ai assisted planning, draft grouping, code and documentation. The user executed and reviewed the work. References are SAM-derived silver masks, not independent gold.</p>
<h2>Dataset and split</h2><p>110 originals; 88 development / 22 held-out evaluation; zero declared visual-family crossings. Evaluation groups: {html.escape(', '.join(split.get('eval_groups', [])) if split else 'PENDING')}. The conservative 58-image kitchen family remains development-only, limiting held-out multi-egg/scene diversity. Visual grouping cannot prove physical-egg/session independence.</p>
<h2>Methods</h2><p>Pretrained SAM ViT-B with recorded per-egg prompts and human review created visible-egg reference masks; no SAM training. Classical inference used only original pixels and the selected configuration. {len(trials)} configurations were compared on DEV only; {html.escape(selected.get('selected_config_id','PENDING') if selected else 'PENDING')} was selected by macro DEV IoU. The algorithm was frozen before held-out reference pixels were read.</p>
<h2>Metrics</h2><table><tr><th>Split</th><th>IoU</th><th>Dice</th><th>Precision</th><th>Recall</th></tr>{metric_html}</table>
<p>These are agreement scores against model-derived, human-reviewed references—not absolute accuracy against independent human annotations.</p>
<h2>Classical DEV trials</h2><table><tr><th>Config</th><th>Method</th><th>IoU</th><th>Dice</th><th>Precision</th><th>Recall</th><th>Errors</th></tr>{trial_html}</table>
<h2>Held-out qualitative cases</h2><table><tr><th>Image</th><th>Group</th><th>IoU</th><th>Dice</th></tr>{worst_html}</table>{images_html}
<h2>Full-video tracking</h2><p>Status: {html.escape(video.get('status','PENDING') if video else 'PENDING')}; frames: {video.get('frames_processed','PENDING') if video else 'PENDING'}/{video.get('frames','PENDING') if video else 'PENDING'}; tracker: {html.escape(video.get('actual_tracker','PENDING') if video else 'PENDING')}; labels: {html.escape(', '.join(video.get('object_ids',[])) if video else 'PENDING')}; not-detected object-frames: {video.get('lost_object_frames','PENDING') if video else 'PENDING'}; out-of-scene object-frames: {video.get('out_of_scene_object_frames','not recorded') if video else 'PENDING'}. {html.escape(tracking_note)} Output has no audio. Selected file: {html.escape(video_info.get('file_name', 'PENDING'))}; SHA-256: {html.escape(video_info.get('sha256', 'PENDING'))}. Frame counts alone do not establish visual correctness or tracking accuracy.</p>
<h2>Public submission links</h2><p><a href="{html.escape(video_url, quote=True)}">Public annotated Drive video</a></p><p><a href="{html.escape(github_url, quote=True)}">Public GitHub repository</a></p>
<h2>Reproducibility and limitations</h2><pre>{html.escape(commands)}</pre><p>See REPORT.md for the complete methodology, configuration, limitations and checklist. Public GitHub and Drive links must be added after upload.</p>
</body></html>"""
    (report_dir / "REPORT.html").write_text(page)

    checklist = f"""# Final submission checklist

- [ ] `REPORT.pdf` / `REPORT.md` / `REPORT.html` claims match the actual outputs and overlays.
- [ ] `report/REPORT.pdf` is present in GitHub and contains clickable public Drive-video and GitHub links.
- [ ] Inspect the actual qualitative panels; describe only failure behavior that is visible.
- [ ] The public Drive URL opens **{video_info.get('file_name', 'the accepted corrected MP4')}**, not the rejected template output.
- [ ] Drive sharing is **Anyone with the link — Viewer**; test in an incognito/logged-out window.
- [ ] Follow `report/GITHUB_PUSH.md` to push the generated `repo/` contents to {github_url}.
- [ ] Confirm scripts 01–10, split freeze, prompts, all 16 DEV trials, final metrics and video tracks are present.
- [ ] Do not upload the 375 MB checkpoint, raw dataset, source video or annotated MP4 to GitHub.
- [ ] Keep the optional live-demo script out of this submission; it is not part of the accepted pipeline.
- [ ] Retain AI-use disclosure, silver-reference limitation and tracking limitations.
- [ ] Test the public GitHub repository and both PDF links while logged out/incognito.
- [ ] Submit within seven days of receiving the assignment; retain the original receipt/deadline evidence.
"""
    (report_dir / "SUBMISSION_CHECKLIST.md").write_text(checklist)
    readme = f"""# Egg CV — Segmentation and Tracking

Local result status: **{status}**.

Pretrained-SAM reference annotations, DEV-only classical experiments, a frozen held-out evaluation and full-video individual tracking. No SAM training is claimed. References are human-reviewed model-derived silver masks, not independent gold.

## Reviewer quick path — no execution needed

1. Open **`report/REPORT.pdf`** for methods, metrics, limitations and public links.
2. Open the **public annotated Drive video**: {video_url}
3. Inspect `results/`, `experiments/`, `references/` and the saved evaluation overlays.
4. Public repository: {github_url}

A reviewer does **not** need to repeat the 110-image manual annotation to study this submission.

## Requirement-to-evidence map

| Assignment requirement | Repository evidence |
|---|---|
| Task 1 — pretrained segmentation | `scripts/05_setup_sam.py`, `scripts/06_sam_widget.py`, `references/` |
| Task 2 — 4:1 split and classical trials | `metadata/split*`, `scripts/07_classical_dev_experiments.py`, `experiments/` |
| Task 3 — quantitative and qualitative evaluation | `scripts/08_classical_final_eval.py`, `results/`, `predictions/classical/eval_overlays/` |
| Task 4 — individual video tracking | `scripts/09_track_full_video.py`, `results/video_tracks.csv`, `results/video_tracking_summary.json`, public Drive MP4 |
| Submission report and reproduction | `report/REPORT.pdf`, `report/REPRODUCE.md`, `requirements.txt` |

## Clone the repository

```bash
git clone {github_url}
cd {Path(github_url.rstrip('/')).name}
```

## Reproduce in Google Colab — recommended

1. Create a Colab notebook and mount Google Drive.
2. Use project root `/content/drive/MyDrive/Egg_CV_Task`; put code in `scripts/`, original images under `data/raw/`, and the authorized video at `data/video/input.mp4`.
3. Follow **`report/REPRODUCE.md`** for exact numbered commands and environment notes.
4. Use a GPU for the fastest SAM candidate generation. The CPU box-only fallback is documented for GPU-quota interruptions. Classical experiments, evaluation, tracking and packaging can run on CPU.
5. Saved references and frozen split are included for inspection. Full-from-scratch annotation is optional; do not rerun or resplit merely to obtain different metrics.

## Repository map

- `scripts/` — numbered pipeline and shared utilities
- `metadata/` — manifest, approved groups, frozen split and environment
- `references/` — reviewed masks, instances and per-egg prompt records
- `experiments/` — all DEV trials, failures and selection record
- `results/` — held-out metrics and full-video tracks/summary
- `report/` — PDF/Markdown/HTML report and reproduction instructions

Raw images, the 375 MB checkpoint and large videos are intentionally excluded from GitHub.

## AI disclosure

Arena.ai assisted planning, draft visual grouping, code and documentation. The user executed, reviewed and remains responsible for verification.
"""
    (report_dir / "README.md").write_text(readme)
    reproduce = f"""# Reproduce the experiments in Colab

## Reviewer note

You can review the PDF, public video, saved metrics, masks and prompt records without executing anything. Full manual re-annotation is optional. Do not rerun the split or final EVAL merely to obtain different numbers.

## 1. Clone and install

Open a fresh Google Colab notebook. A T4 GPU is recommended for SAM candidate generation; CPU is sufficient for classical experiments, EVAL, detection tracking and packaging.

```python
!git clone {github_url} /content/egg_cv_repo
!pip install -r /content/egg_cv_repo/requirements.txt
```

Keep Colab's compatible preinstalled torch/torchvision build where possible. The SAM setup script pins the official implementation commit and verifies the checkpoint SHA-256. Exact executed environment details are in `metadata/sam_environment.json`.

## 2. Prepare the Drive project

```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/Egg_CV_Task
!cp -r /content/egg_cv_repo/scripts /content/drive/MyDrive/Egg_CV_Task/
```

Use project root `/content/drive/MyDrive/Egg_CV_Task`. Place the original 110 images under `data/raw/` as documented by the manifest script, and the authorized full source video at `data/video/input.mp4`. Raw media and the approximately 375 MB checkpoint are intentionally not stored in GitHub.

Input sources and expected structure:

- Dataset page: <https://universe.roboflow.com/fire-bf7qe/egg-vzhtr/dataset/1>
- Alternative dataset ZIP: <https://drive.google.com/file/d/1a58rACcCFnaj0ZiDoNyGbDWXunRyWziv/view?usp=sharing>
- Step 01 asks for the extracted dataset folder containing `train/`, `valid/` and `test/`; it copies that structure to `data/raw/` and creates the hash manifest.
- Original video reference: <https://www.youtube.com/watch?v=Y4JyV94y2_A>. Use your authorized full MP4 copy at `data/video/input.mp4`; the source video is not redistributed in this repository.

## 3. Numbered pipeline

```python
{commands}
```

### What each step does

1. **01 — Manifest:** audit the original images, hashes, dimensions and duplicate status.
2. **02 — Contact sheets:** create visual audit sheets; do not treat them as annotations.
3. **03 — Scene groups:** create/review conservative visual families for leakage reduction.
4. **04 — Freeze split:** freeze whole groups into 88 DEV and 22 EVAL images. Never reroll based on metrics.
5. **05 — SAM setup:** install/pin pretrained SAM ViT-B and verify the checkpoint; no training occurs.
6. **06 — Reference review:** create human-reviewed SAM-derived masks with per-instance prompts. This is interactive inference and review, not independent gold annotation or model training.
7. **07 — DEV experiments:** run all 16 fixed classical configurations on DEV only, log every trial and select by macro DEV IoU with declared tie-breaks.
8. **08 — Frozen EVAL:** freeze code/config/reference identities before reading EVAL reference pixels, then evaluate the 22 held-out images once. Do not tune afterward.
9. **09 — Full-video tracking:** use detection-gated tracking with persistent human labels at segment starts. Review the entire annotated MP4 visually; completion counts are not tracking accuracy.
10. **10 — Submission package:** generate the linked PDF, reports, repository folder and ZIP from actual saved artifacts. Public link accessibility still requires a manual logged-out test.

## GPU-quota fallback

For Step 06 in a fresh CPU runtime, install the pinned SAM package and run:

```python
%run scripts/06_sam_widget.py --cpu --boxes-only
```

This uses the same checkpoint and reference policy with manual boxes. Accepted images resume from Drive.

## Outputs to inspect

- `references/review_status.csv` and `references/prompts.jsonl`
- `experiments/classical_dev_summary.csv` and `classical_dev_per_image.csv`
- `results/classical_eval_summary.json` and `classical_eval_per_image.csv`
- `predictions/classical/eval_overlays/`
- `results/video_tracks.csv` and `video_tracking_summary.json`
- `report/REPORT.pdf`

## Integrity and media policy

Do not reroll the frozen split, pass SAM/reference prompts to classical inference, or tune on EVAL. Keep unsuccessful DEV trials in the log. The annotated video is linked from Google Drive because the assignment explicitly requires that public Drive link in the PDF; large videos are not duplicated in GitHub. Raw images, checkpoint and large videos belong on Drive, not GitHub.
"""
    (report_dir / "REPRODUCE.md").write_text(reproduce)
    push_guide = f"""# Push the generated repository to GitHub

The target public repository is: {github_url}

## Recommended: Git Bash on Windows

1. Download `submission/Egg_CV_Repository.zip` from Drive and extract it to an empty folder.
2. Open that extracted folder. It must directly contain `README.md`, `scripts/`, `report/`, `metadata/`, etc.—not a second nested repository folder.
3. Install Git for Windows if needed, then open **Git Bash** in the extracted folder.
4. Run:

```bash
git init
git add .
git status
git commit -m "Complete egg segmentation and tracking assignment"
git branch -M main
git remote add origin {github_url}
git push -u origin main
```

If Git asks for your identity, set your own name and GitHub email:

```bash
git config --global user.name "YOUR NAME"
git config --global user.email "YOUR GITHUB EMAIL"
```

Use the normal Git Credential Manager/browser sign-in prompt. Never paste a password or personal access token into chat, the report or the repository. Do not use force-push. These commands assume the public repository was created empty; if it already has conflicting commits, stop and resolve that safely.

## After pushing

- Open {github_url} in an incognito/logged-out browser.
- Confirm `report/REPORT.pdf` exists and opens.
- Open the PDF's Drive-video link and confirm it plays the accepted corrected MP4 without sign-in.
- Confirm `README.md`, `report/REPRODUCE.md`, `requirements.txt`, scripts 01–10, all 16 trial logs and EVAL outputs are visible.
- Submit the public repository URL within the assignment deadline.
"""
    (report_dir / "GITHUB_PUSH.md").write_text(push_guide)
    requirements = ["# Use Colab's compatible preinstalled torch/torchvision; exact versions are in metadata/sam_environment.json",
                    "numpy", "pandas", "Pillow", "requests", "matplotlib", "opencv-python", "ipywidgets", "reportlab", "pypdf",
                    f"segment-anything @ git+https://github.com/facebookresearch/segment-anything.git@{sam.get('sam_commit','dca509fe793f601edb92606367a655c15ac00fdf') if sam else 'dca509fe793f601edb92606367a655c15ac00fdf'}"]
    (report_dir / "requirements.txt").write_text("\n".join(requirements) + "\n")

    pdf_path = build_pdf(report_dir / "REPORT.pdf", project, {
        "status": status, "generated": now(), "split": split, "sam": sam,
        "reviewed": reviewed, "selected": selected, "trials": trials,
        "evaluation": evaluation, "eval_rows": eval_rows, "video": video,
        "video_url": video_url, "github_url": github_url, "commands": commands,
        "video_info": video_info,
    })
    pdf_qa = verify_final_pdf(pdf_path, video_url, github_url) if final else {"pages": "draft", "uri_annotations": "not_checked"}
    (report_dir / "tracking_video_evidence.json").write_text(json.dumps(video_info, indent=2) + "\n")

    repo = project / "repo"
    if repo.exists():
        shutil.rmtree(repo)
    repo.mkdir()
    (repo / "scripts").mkdir(exist_ok=True)
    for script_name in SCRIPT_FILES:
        script = SCRIPT_DIR_GLOBAL / script_name
        if not script.is_file():
            raise FileNotFoundError(f"Required pipeline script is missing: {script}")
        shutil.copy2(script, repo / "scripts" / script.name)
    # Explicitly exclude the deferred optional live demo and any scratch scripts.
    for filename in ("manifest.csv", "scene_groups.csv", "split.csv", "pilot_images.csv", "split_freeze.json",
                     "scene_group_policy_draft.json", "sam_environment.json", "classical_eval_freeze.json"):
        path = metadata / filename
        if path.exists():
            (repo / "metadata").mkdir(exist_ok=True)
            shutil.copy2(path, repo / "metadata" / filename)
    for folder in ("experiments", "results"):
        copy_tree_contents(project / folder, repo / folder)
        # Large media stays on Drive, not GitHub.
        for media in (repo / folder).rglob("*.mp4") if (repo / folder).exists() else []:
            media.unlink()
    for folder in ("masks", "instances", "annotations"):
        copy_tree_contents(project / "references" / folder, repo / "references" / folder)
    for filename in ("prompts.jsonl", "review_status.csv", "reference_policy.json"):
        path = project / "references" / filename
        if path.exists():
            (repo / "references").mkdir(exist_ok=True)
            shutil.copy2(path, repo / "references" / filename)
    for folder in ("dev", "eval"):
        copy_tree_contents(project / "predictions" / "classical" / folder, repo / "predictions" / "classical" / folder)
    copy_tree_contents(report_dir, repo / "report")
    shutil.copy2(report_dir / "README.md", repo / "README.md")
    shutil.copy2(report_dir / "requirements.txt", repo / "requirements.txt")
    (repo / ".gitignore").write_text("checkpoints/\ndata/raw/\ndata/video/\n*.part\n*.pth\n*.mp4\n.DS_Store\n__pycache__/\n")
    (repo / "PUBLIC_LINKS.md").write_text(f"# Public links\n\n- GitHub: {github_url}\n- Annotated video: {video_url}\n")
    required_repo_files = [repo / "README.md", repo / "requirements.txt", repo / "report" / "REPORT.pdf",
                           repo / "report" / "REPRODUCE.md", repo / "report" / "GITHUB_PUSH.md",
                           repo / "results" / "video_tracks.csv"]
    absent = [str(path.relative_to(repo)) for path in required_repo_files if not path.is_file()]
    if final and absent:
        raise RuntimeError(f"Repository is missing required evidence: {absent}")
    if (repo / "scripts" / "11_live_tracking_demo.py").exists():
        raise RuntimeError("Deferred live-demo script must not enter the submission repository.")
    manifest_rows = []
    for path in sorted(repo.rglob("*")):
        if path.is_file():
            if path.suffix.lower() == ".mp4":
                raise RuntimeError(f"Large video must remain on Drive, not GitHub: {path}")
            if path.stat().st_size > 95 * 1024 * 1024:
                raise RuntimeError(f"File exceeds the safe GitHub size threshold (95 MiB): {path}")
            manifest_rows.append({"relative_path": path.relative_to(repo).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    with (repo / "submission_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "bytes", "sha256"], lineterminator="\n")
        writer.writeheader(); writer.writerows(manifest_rows)
    submission = project / "submission"
    submission.mkdir(exist_ok=True)
    zip_path = submission / "Egg_CV_Repository.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(repo.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(repo))
    print("\nREPORT/PACKAGE STATUS:", status)
    if missing:
        print("Still missing:", ", ".join(missing))
    print("PDF report:", pdf_path)
    print("HTML report:", report_dir / "REPORT.html")
    print("Repository folder:", repo)
    print("Repository ZIP:", zip_path)
    print("Repository ZIP bytes:", zip_path.stat().st_size)
    print("Files in repository:", len(manifest_rows))
    print("PDF QA:", pdf_qa)
    print("Annotated video evidence:", json.dumps(video_info, indent=2))
    print("Next: extract the ZIP, follow report/GITHUB_PUSH.md, push to the public repo, and test GitHub + both PDF links incognito.")
    print("Links are embedded and checked in the PDF. PUBLIC UPLOAD/ACCESSIBILITY IS STILL A MANUAL CHECK; no upload is claimed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default="/content/drive/MyDrive/Egg_CV_Task")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--video-link", default="", help="Public Drive viewer URL for the annotated tracking video")
    parser.add_argument("--github-link", default="", help="Public GitHub repository URL")
    args = parser.parse_args()
    main(args.project_root, args.final, args.video_link, args.github_link)
