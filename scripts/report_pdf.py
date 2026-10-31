"""Create the submission PDF from real pipeline artifacts using ReportLab.

Called by 10_build_submission.py. No metrics are invented or read from eval for
algorithm tuning. Developed with Arena.ai assistance; review before submission.
"""

from pathlib import Path
import html
import json
import subprocess
import sys


def _reportlab():
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                        TableStyle, Image, PageBreak, Preformatted)
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "reportlab"])
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                        TableStyle, Image, PageBreak, Preformatted)
    return (colors, TA_CENTER, A4, getSampleStyleSheet, ParagraphStyle, inch,
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, Preformatted)


def build_pdf(output_path, project, context):
    (colors, TA_CENTER, A4, getSampleStyleSheet, ParagraphStyle, inch,
     SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
     Preformatted) = _reportlab()
    project = Path(project)
    output_path = Path(output_path)
    split = context["split"] or {}
    sam = context["sam"] or {}
    selected = context["selected"] or {}
    evaluation = context["evaluation"] or {}
    video = context["video"] or {}
    dev = selected.get("dev_metrics", {})
    ev = evaluation.get("eval_macro", {})
    micro = evaluation.get("eval_micro", {})
    trials = context["trials"]
    eval_rows = context["eval_rows"]
    video_url = context["video_url"]
    github_url = context["github_url"]
    video_info = context.get("video_info", {})
    object_accounting = "; ".join(
        f"{label}: tracked={stats.get('tracked', 0)}, initialized={stats.get('initialized', 0)}, "
        f"not_detected={stats.get('not_detected', 0)}, out_of_scene={stats.get('out_of_scene', 0)}"
        for label, stats in sorted(video.get("object_stats", {}).items())
    ) or "PENDING"

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold",
                              fontSize=22, leading=27, textColor=colors.HexColor("#14395f"), spaceAfter=14))
    styles.add(ParagraphStyle(name="H", parent=styles["Heading2"], fontName="Helvetica-Bold",
                              fontSize=14, leading=18, textColor=colors.HexColor("#14395f"),
                              spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="Body", parent=styles["BodyText"], fontName="Helvetica",
                              fontSize=9.3, leading=13.5, spaceAfter=6))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontName="Helvetica",
                              fontSize=7.6, leading=10.5, textColor=colors.HexColor("#45566e"), spaceAfter=5))
    styles.add(ParagraphStyle(name="CodeBlock", parent=styles["Code"], fontName="Courier",
                              fontSize=6.8, leading=9, leftIndent=8, rightIndent=8,
                              backColor=colors.HexColor("#f1f4f8"), borderPadding=7))
    styles.add(ParagraphStyle(name="Link", parent=styles["BodyText"], fontName="Helvetica-Bold",
                              fontSize=9.5, leading=13, textColor=colors.HexColor("#0759b8"), spaceAfter=6))

    def p(text, style="Body"):
        return Paragraph(text, styles[style])

    def link(label, url):
        safe = html.escape(url, quote=True)
        return p(f'<link href="{safe}" color="#0759b8">{html.escape(label)}</link>', "Link")

    def fmt(value):
        return f"{float(value):.4f}" if value is not None else "PENDING"

    def table(data, widths=None, font=7.6):
        result = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        result.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe9f5")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#14395f")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), font),
            ("LEADING", (0, 0), (-1, -1), font + 2.4),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#bac8d9")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fc")]),
        ]))
        return result

    story = [
        p("Egg Segmentation and Individual Tracking", "ReportTitle"),
        p(f"<b>Status:</b> {html.escape(context['status'])} | <b>Generated UTC:</b> {html.escape(context['generated'])}"),
        p("<b>Reviewer quick path:</b> read this report, inspect the linked annotated video and view the saved metrics/overlays. Re-annotating 110 images is not required to review the submission. Full Colab reproduction instructions are in README.md and report/REPRODUCE.md."),
        link("Open public annotated-video output", video_url),
        link("Open public GitHub repository", github_url),
        p("<b>AI-use disclosure:</b> Arena.ai assisted planning, draft visual grouping, code and documentation. The user executed the pipeline, selected/reviewed prompts and remains responsible for verification. No unexecuted result is claimed."),
        p("Assignment requirement-to-evidence map", "H"),
        table([["Requirement", "Evidence in this submission"],
               ["Task 1 — pretrained segmentation", "Steps 05–06; pretrained SAM ViT-B; 110 reviewed references; prompts, masks and review status."],
               ["Task 2 — 4:1 split and classical trials", "Steps 03–04 and 07; frozen 88/22 group split; all 16 DEV configurations and selection log."],
               ["Task 3 — quantitative and qualitative evaluation", "Step 08; frozen EVAL metrics, per-image counts, worst-case table and actual comparison panels."],
               ["Task 4 — individual video tracking", "Step 09; persistent raw/boiled IDs, full-video CSV/JSON and public corrected Drive MP4."],
               ["Submission and reproduction", "This PDF in GitHub; clickable public links; README, REPRODUCE, requirements and scripts 01–10."]],
              [1.85*inch, 5.05*inch], 7.2),
        p("1. Task and dataset", "H"),
        p("The supplied Roboflow v1 classification dataset contains 110 original 640 x 640 JPG images but no independent pixel-level gold masks. The work covers pretrained-SAM reference creation, a separate classical segmentation method, group-aware development/evaluation and individual egg tracking on the full supplied video."),
        p("<b>Reference limitation:</b> the masks are human-reviewed SAM-derived silver references. Classical scores are agreement with these references, not absolute accuracy against independent human ground truth."),
        p("2. Leakage-reduction split", "H"),
        p(f"Related visual setups were grouped before experiments and frozen into 88 development and 22 held-out evaluation images. Declared group crossings: 0. Evaluation groups: {html.escape(', '.join(split.get('eval_groups', [])))}."),
        p("The conservative 58-image kitchen family G11, including the clearly visible multi-egg scenes, remains development-only. Held-out multi-egg and scene diversity are therefore limited. Visual families reduce obvious setup leakage but cannot prove unique people, physical eggs or capture sessions without provenance."),
        p("3. Pretrained SAM reference creation", "H"),
        p(f"Pretrained SAM ViT-B was used without training or fine-tuning. Official commit: {html.escape(sam.get('sam_commit', 'PENDING'))}. Reviewed references: {context['reviewed']}/110. Automatic grid candidates or manual boxes supported prompt discovery; accepted instances record point+box or human-box prompts and human review decisions."),
        p(f"Checkpoint SHA-256: {html.escape(sam.get('checkpoint', {}).get('sha256', 'PENDING'))}", "Small"),
        p("The label policy marks visible egg pixels only, excludes hand/background and does not invent hidden regions. Empty images use explicitly confirmed empty references. Binary union, instance masks, prompts and review status are retained."),
        p("4. Classical development experiments", "H"),
        p(f"Classical inference receives only original pixels and a configuration, never SAM/reference prompts or boxes. {len(trials)} fixed configurations using HSV/Lab seeds, morphology, connected components and/or GrabCut were compared on DEV only. Selection used maximum macro DEV mean IoU, then mean Dice and micro IoU tie-breaks."),
        p(f"Selected: <b>{html.escape(selected.get('selected_config_id', 'PENDING'))}</b>. The method was frozen before held-out reference pixels were read."),
    ]
    config_text = json.dumps(selected.get("config", {}), indent=2)
    story += [Preformatted(config_text, styles["CodeBlock"]), Spacer(1, 8)]

    story += [p("5. Segmentation results", "H")]
    metrics = [["Split / aggregation", "IoU", "Dice", "Precision", "Recall"],
               ["DEV macro", fmt(dev.get("mean_iou")), fmt(dev.get("mean_dice")), fmt(dev.get("mean_precision")), fmt(dev.get("mean_recall"))],
               ["EVAL macro", fmt(ev.get("mean_iou")), fmt(ev.get("mean_dice")), fmt(ev.get("mean_precision")), fmt(ev.get("mean_recall"))],
               ["EVAL micro", fmt(micro.get("iou")), fmt(micro.get("dice")), fmt(micro.get("precision")), fmt(micro.get("recall"))]]
    story += [table(metrics, [1.65*inch, 0.85*inch, 0.85*inch, 0.95*inch, 0.85*inch], 8.2),
              p("<b>Definitions:</b> IoU = TP/(TP+FP+FN); Dice = 2TP/(2TP+FP+FN); precision = TP/(TP+FP); recall = TP/(TP+FN). Macro averages per-image values with equal image weight. Micro pools pixel counts before calculating the metric. These are overlap measures, not classification accuracy."),
              p(f"EVAL median IoU: <b>{fmt(ev.get('median_iou'))}</b>. Empty policy: both prediction and reference empty scores 1; other zero-division cases score 0. Per-image TP/FP/FN/TN, masks and overlays are saved. EVAL was not used for parameter tuning.")]

    story += [p("Development trial log", "H")]
    trial_data = [["Config", "Method", "IoU", "Dice", "Precision", "Recall", "Errors"]]
    for row in sorted(trials, key=lambda r: float(r["mean_iou"]), reverse=True):
        trial_data.append([row["config_id"], row["method"], fmt(row["mean_iou"]), fmt(row["mean_dice"]),
                           fmt(row["mean_precision"]), fmt(row["mean_recall"]), row["errors"]])
    story += [table(trial_data, [1.25*inch, 1.0*inch, 0.65*inch, 0.65*inch, 0.75*inch, 0.65*inch, 0.55*inch], 6.8),
              p("The grid compared HSV/Lab appearance seeds, warm-color inclusion, GrabCut rectangle sizes, hybrid refinement, component limits and iteration counts. Non-selected trials remain visible rather than being erased. Complete parameter JSON, micro metrics, runtime and per-image rows are retained in experiments/classical_dev_summary.csv and classical_dev_per_image.csv.", "Small")]

    worst = sorted(eval_rows, key=lambda r: float(r["iou"]))[:10]
    story += [p("Lowest-IoU held-out cases", "H")]
    worst_data = [["Image", "Group", "IoU", "Dice", "Pred. area", "Ref. area"]]
    for row in worst:
        worst_data.append([row["image_id"], row["scene_group_id"], fmt(row["iou"]), fmt(row["dice"]),
                           row["prediction_area"], row["reference_area"]])
    if len(worst_data) > 1:
        story += [table(worst_data, [0.7*inch, 0.65*inch, 0.65*inch, 0.65*inch, 1.0*inch, 1.0*inch], 7.2),
                  p("The following saved panels show classical prediction, reference and error pixels. Failure causes must be assigned only after visual inspection, not inferred from the score alone.")]
    for row in worst[:3]:
        path = project / "predictions" / "classical" / "eval_overlays" / f"{row['image_id']}.jpg"
        if path.exists():
            story += [p(f"{row['image_id']} - IoU {fmt(row['iou'])}", "Small"),
                      Image(str(path), width=7.05*inch, height=2.35*inch), Spacer(1, 8)]

    if video.get("actual_tracker") == "detection_gated_hsv":
        tracking_note = ("Detection-gated tracking uses HSV/contour candidates, shape/size gates, one-to-one temporal association and green-card suppression. "
                         "not_detected and out_of_scene rows have blank coordinates; boxes are hidden rather than drifted or interpolated.")
    else:
        tracking_note = "User-assigned boxes at candidate segment starts preserve identities across cuts. Lost tracks are marked rather than interpolated."
    story += [PageBreak(), p("6. Full-video individual tracking", "H"),
              p(f"Status: {video.get('status', 'PENDING')}. Frames processed: {video.get('frames_processed', 'PENDING')} / {video.get('frames', 'PENDING')}. Tracker: {html.escape(str(video.get('actual_tracker', 'PENDING')))}. Persistent labels: {html.escape(', '.join(video.get('object_ids', [])))}."),
              p(f"Candidate segments: {video.get('candidate_segments', 'PENDING')}; not-detected object-frames: {video.get('lost_object_frames', 'PENDING')}; out-of-scene object-frames: {video.get('out_of_scene_object_frames', 'not recorded')}. {tracking_note} OpenCV visualization has no audio. Execution counts alone do not establish visual correctness or measured tracking accuracy."),
              p(f"<b>Per-object status accounting:</b> {html.escape(object_accounting)}. Status counts describe visibility handling; no frame-by-frame tracking ground truth or accuracy percentage is claimed."),
              p(f"<b>Selected annotated file:</b> {html.escape(video_info.get('file_name', 'PENDING'))}; verified frames: {video_info.get('frames', 'PENDING')}; FPS: {video_info.get('fps', 'PENDING')}; resolution: {video_info.get('width', 'PENDING')} x {video_info.get('height', 'PENDING')}. SHA-256: {html.escape(video_info.get('sha256', 'PENDING'))}. The Drive URL must open this same accepted file."),
              p("Cut detection and tracking may fail under occlusion, abrupt motion or ambiguous identity. The complete rendered video must be visually reviewed and observed issues disclosed. Raw/boiled labels come from the video sequence; the tracker is not a learned raw/boiled classifier."),
              link("Open the publicly accessible Drive tracking video", video_url),
              p("7. Reproducibility and repository setup", "H"),
              p("Recommended reproduction environment: Google Colab. Use a GPU for full SAM candidate generation, or the documented CPU box-only fallback. Classical experiments, final evaluation, tracking and packaging run on CPU. Reviewers can inspect saved references and results without repeating the 110-image manual review."),
              Preformatted(context["commands"], styles["CodeBlock"]),
              p("The original dataset and authorized video must be placed at the documented project paths. Raw images, the 375 MB checkpoint and large videos are excluded from GitHub; relative paths and hashes are recorded. Exact package/GPU information is in metadata/sam_environment.json."),
              p("8. Limitations and integrity", "H"),
              p("1. SAM-derived references are not independent gold and may contain review errors.<br/>2. Visual grouping is conservative and cannot guarantee session or physical-egg independence.<br/>3. Whole-group splitting limits held-out scene and multi-egg diversity.<br/>4. DEV model selection can overfit; held-out results were not used for further tuning.<br/>5. Video tracking uses manual segment-start boxes and explicit loss records, not a learned raw/boiled classifier.<br/>6. AI assistance is disclosed; code, annotations and outputs require human verification."),
              p("9. Public submission links", "H"),
              link("Public annotated-video output", video_url),
              link("Public GitHub repository", github_url),
              p("Public accessibility must be tested in an incognito/logged-out browser before submission. The annotated MP4 is intentionally kept on Drive rather than duplicated in GitHub because the assignment explicitly requires a public Drive link in this PDF. See report/GITHUB_PUSH.md and report/REPRODUCE.md for push and experiment instructions.")]

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#5a687b"))
        canvas.drawString(0.6*inch, 0.38*inch, "Egg CV - execution report")
        canvas.drawRightString(A4[0] - 0.6*inch, 0.38*inch, f"Page {document.page}")
        canvas.restoreState()

    document = SimpleDocTemplate(str(output_path), pagesize=A4,
                                 rightMargin=0.55*inch, leftMargin=0.55*inch,
                                 topMargin=0.55*inch, bottomMargin=0.6*inch,
                                 title="Egg Segmentation and Individual Tracking",
                                 author="Egg CV assignment")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    if not output_path.exists() or output_path.stat().st_size < 1000:
        raise RuntimeError("PDF generation did not produce a valid nonempty file.")
    return output_path
