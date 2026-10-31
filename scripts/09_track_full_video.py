"""Track individual eggs through the FULL video with persistent user labels.

Place the authorized source video at data/video/input.mp4 (or pass --video).
Pass 1 detects possible cuts. At each segment start, use the visible Boxes widget:
  raw:100,180,220,340;boiled:390,180,510,340
SAVE BOX & NEXT stores each segment. COPY PREVIOUS, NO VISIBLE EGG, BACK/EDIT
and SAVE & STOP are available. After the last segment, tracking runs automatically.

Recommended --tracker detection: HSV/contour egg candidates, shape/size gates,
temporal per-ID association and green-card suppression. It reports missed or
out-of-scene objects instead of drifting a template onto hands/text/background.
Legacy CSRT/KCF/MOSSE and template_ncc modes remain available for comparison.
Cuts are manually re-initialized with the SAME labels. No automatic raw/boiled
recognition or tracker training is claimed. Output visualization has no audio.

Developed with Arena.ai assistance. Inspect the actual video and disclose limits.
"""

from pathlib import Path
from collections import deque
import argparse
from datetime import datetime, timezone
import csv
import json
import math
import re
import shutil
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import ipywidgets as widgets

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from classical_utils import file_sha256

COLORS = {"raw": (60, 180, 60), "boiled": (60, 100, 240)}
FALLBACK_COLORS = [(230, 160, 40), (180, 70, 200), (50, 180, 180), (220, 70, 140)]


def now():
    return datetime.now(timezone.utc).isoformat()


def frame_signature(frame):
    small = cv2.resize(frame, (192, 108), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return gray, hist


def scan_video(video_path, hist_threshold=0.25, mad_threshold=22.0, hard_mad=55.0):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Cannot open video. Check the path and codec.")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    previous, starts, reasons = None, [0], {}
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray, hist = frame_signature(frame)
        if previous is not None and index > 0:
            old_gray, old_hist = previous
            mad = float(np.mean(cv2.absdiff(gray, old_gray)))
            correlation = float(cv2.compareHist(old_hist, hist, cv2.HISTCMP_CORREL))
            is_cut = (correlation < hist_threshold and mad > mad_threshold) or mad > hard_mad
            if is_cut and (index - starts[-1]) >= 5:
                starts.append(index)
                reasons[str(index)] = {"histogram_correlation": correlation, "mean_absolute_difference": mad}
        previous = (gray, hist)
        index += 1
        if index % 1000 == 0:
            print(f"Scene scan: {index} frames", flush=True)
    cap.release()
    if index == 0:
        raise RuntimeError("Video contains no readable frames.")
    segments = []
    for number, start in enumerate(starts):
        end = starts[number + 1] if number + 1 < len(starts) else index
        segments.append({"segment_id": f"S{number:03}", "start_frame": start, "end_frame": end,
                         "start_seconds": start / fps, "end_seconds": end / fps})
    return {"fps": fps, "width": width, "height": height, "frames": index,
            "segments": segments, "cut_reasons": reasons,
            "detector": {"hist_threshold": hist_threshold, "mad_threshold": mad_threshold,
                         "hard_mad": hard_mad, "note": "Candidate cuts only; user confirms boxes at each segment."}}


def read_frame(video_path, frame_number):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_number}")
    return frame


def parse_labeled_boxes(text, width, height):
    annotations = []
    labels = set()
    text = text.replace("；", ";").replace("，", ",")
    for item in text.split(";"):
        if not item.strip():
            continue
        match = re.fullmatch(r"\s*([A-Za-z][A-Za-z0-9_-]*)\s*:\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*", item)
        if not match:
            raise ValueError("Use label:x1,y1,x2,y2;label2:x1,y1,x2,y2")
        label = match.group(1)
        if label in labels:
            raise ValueError("Duplicate label")
        labels.add(label)
        x1, y1, x2, y2 = [float(match.group(i)) for i in range(2, 6)]
        x1, x2 = sorted((max(0, min(width - 1, x1)), max(0, min(width - 1, x2))))
        y1, y2 = sorted((max(0, min(height - 1, y1)), max(0, min(height - 1, y2))))
        if x2 - x1 < 5 or y2 - y1 < 5:
            raise ValueError("Box too small")
        annotations.append({"object_id": label, "box_xyxy": [x1, y1, x2, y2]})
    if not annotations:
        raise ValueError("No boxes; use none if no eggs are visible")
    return annotations


def annotate_segments(video_path, scan, work):
    path = work / "segment_annotations.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["video_sha256"] != scan["video_sha256"] or saved["scan_sha256"] != scan["scan_sha256"]:
            raise RuntimeError("Existing segment annotations belong to another video/scan.")
    else:
        saved = {"video_sha256": scan["video_sha256"], "scan_sha256": scan["scan_sha256"], "annotations": {}}
    previous = None
    for index, segment in enumerate(scan["segments"]):
        sid = segment["segment_id"]
        if sid in saved["annotations"]:
            previous = saved["annotations"][sid]
            continue
        frame = read_frame(video_path, segment["start_frame"])
        plt.close("all")
        fig, axis = plt.subplots(figsize=(11, 7))
        axis.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        axis.set_title(f"{sid} | frame {segment['start_frame']} | {segment['start_seconds']:.2f}s\nEnter every visible egg with persistent identity labels")
        axis.grid(alpha=0.3)
        display(fig)
        plt.close(fig)
        print("Format: raw:x1,y1,x2,y2;boiled:x1,y1,x2,y2")
        while True:
            command = input(f"{sid} boxes (copy/none/quit): ").strip()
            if command.lower() == "quit":
                path.write_text(json.dumps(saved, indent=2) + "\n")
                print("Annotation progress saved. Rerun to continue.")
                return None
            try:
                if command.lower() == "none":
                    annotations = []
                elif command.lower() == "copy":
                    if previous is None:
                        raise ValueError("No previous segment to copy")
                    annotations = previous
                else:
                    annotations = parse_labeled_boxes(command, scan["width"], scan["height"])
                saved["annotations"][sid] = annotations
                path.write_text(json.dumps(saved, indent=2) + "\n")
                previous = annotations
                break
            except ValueError as exc:
                print("Error:", exc)
    print(f"All {len(scan['segments'])} segment starts annotated.")
    return saved


class VideoAnnotationUI:
    """Visible widget boxes; on the last segment it triggers full tracking."""

    def __init__(self, video_path, scan, work, on_complete):
        self.video_path = Path(video_path)
        self.scan = scan
        self.work = Path(work)
        self.on_complete = on_complete
        self.path = self.work / "segment_annotations.json"
        if self.path.exists():
            self.saved = json.loads(self.path.read_text())
            if self.saved["video_sha256"] != scan["video_sha256"] or self.saved["scan_sha256"] != scan["scan_sha256"]:
                raise RuntimeError("Existing segment annotations belong to another video/scan.")
        else:
            self.saved = {"video_sha256": scan["video_sha256"], "scan_sha256": scan["scan_sha256"], "annotations": {}}
        self.index = 0
        self.finished = False
        self.busy = False
        self.title = widgets.HTML()
        self.image_out = widgets.Output()
        self.log_out = widgets.Output(layout={"height": "85px", "overflow_y": "auto"})
        self.boxes = widgets.Text(description="Boxes", placeholder="raw:x1,y1,x2,y2;boiled:x1,y1,x2,y2",
                                  layout=widgets.Layout(width="620px"))
        self.save = widgets.Button(description="SAVE BOX & NEXT", button_style="primary")
        self.copy = widgets.Button(description="COPY PREVIOUS")
        self.none = widgets.Button(description="NO VISIBLE EGG")
        self.back = widgets.Button(description="BACK / EDIT")
        self.quit = widgets.Button(description="SAVE & STOP", button_style="danger")
        self.box_row = widgets.HBox([self.boxes, self.save])
        self.nav = widgets.HBox([self.copy, self.none, self.back, self.quit])
        self.root = widgets.VBox([
            self.title, self.image_out, self.box_row, self.nav,
            widgets.HTML("<b>Use ORIGINAL axes. One box per visible egg. Examples: raw:100,180,220,340 or boiled:390,180,510,340. Separate multiple eggs with semicolons.</b>"),
            self.log_out,
        ])
        self.save.on_click(lambda _: self.save_text())
        self.copy.on_click(lambda _: self.copy_previous())
        self.none.on_click(lambda _: self.save_current([]))
        self.back.on_click(lambda _: self.go_back())
        self.quit.on_click(lambda _: self.stop())

    def log(self, text):
        with self.log_out:
            print(text)

    def start(self):
        display(self.root)
        self.show_current()

    def previous_annotation(self):
        if self.index <= 0:
            return None
        previous_id = self.scan["segments"][self.index - 1]["segment_id"]
        return self.saved["annotations"].get(previous_id)

    def show_current(self):
        if self.finished:
            return
        while self.index < len(self.scan["segments"]):
            sid = self.scan["segments"][self.index]["segment_id"]
            if sid not in self.saved["annotations"]:
                break
            self.index += 1
        if self.index >= len(self.scan["segments"]):
            self.complete()
            return
        segment = self.scan["segments"][self.index]
        sid = segment["segment_id"]
        self.title.value = (f"<h3>{sid} | frame {segment['start_frame']} | {segment['start_seconds']:.2f}s</h3>"
                            "<p>Enter every visible egg with its persistent raw/boiled label, then SAVE BOX & NEXT.</p>")
        frame = read_frame(self.video_path, segment["start_frame"])
        with self.image_out:
            clear_output(wait=True)
            plt.close("all")
            fig, axis = plt.subplots(figsize=(11, 7))
            axis.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            axis.set_title(f"{sid} — ORIGINAL frame; use pixel axes for boxes")
            axis.grid(alpha=0.3)
            display(fig)
            plt.close(fig)
        self.boxes.value = ""
        self.box_row.layout.display = "flex"
        self.nav.layout.display = "flex"

    def save_text(self):
        if self.busy or self.finished:
            return
        try:
            annotations = parse_labeled_boxes(self.boxes.value, self.scan["width"], self.scan["height"])
            self.save_current(annotations)
        except ValueError as exc:
            self.log(f"Box error: {exc}. Use raw:x1,y1,x2,y2 or boiled:x1,y1,x2,y2")

    def save_current(self, annotations):
        if self.busy or self.finished:
            return
        sid = self.scan["segments"][self.index]["segment_id"]
        self.saved["annotations"][sid] = annotations
        self.path.write_text(json.dumps(self.saved, indent=2) + "\n")
        self.log(f"Saved {sid}: {len(annotations)} egg box(es).")
        self.index += 1
        self.show_current()

    def copy_previous(self):
        previous = self.previous_annotation()
        if previous is None:
            self.log("No previous segment boxes available to copy.")
            return
        self.save_current(previous)

    def go_back(self):
        if self.busy or self.finished or self.index <= 0:
            return
        self.index -= 1
        sid = self.scan["segments"][self.index]["segment_id"]
        self.saved["annotations"].pop(sid, None)
        self.path.write_text(json.dumps(self.saved, indent=2) + "\n")
        self.show_current()

    def stop(self):
        self.path.write_text(json.dumps(self.saved, indent=2) + "\n")
        self.finished = True
        self.box_row.layout.display = "none"
        self.nav.layout.display = "none"
        self.title.value = "<h3>Annotation progress saved</h3><p>Rerun Step 09 to resume; accepted segment boxes are preserved.</p>"

    def complete(self):
        if self.busy:
            return
        self.busy = True
        self.box_row.layout.display = "none"
        self.nav.layout.display = "none"
        self.title.value = "<h3>All segment boxes saved. Tracking the full video…</h3>"
        try:
            summary = self.on_complete(self.saved)
            self.finished = True
            self.title.value = "<h3>FULL VIDEO TRACKING COMPLETE</h3><p>Review video_tracked.mp4, then create its public Drive viewer link.</p>"
            with self.image_out:
                clear_output(wait=True)
                print(json.dumps({k: summary[k] for k in ("frames_processed", "track_rows", "lost_object_frames", "actual_tracker", "object_ids")}, indent=2))
        except Exception as exc:
            self.title.value = f"<h3>Tracking error</h3><p>{html_escape(str(exc))}</p><p>Rerun Step 09; saved boxes will be reused.</p>"
            self.log(f"Tracking error: {exc}")
        finally:
            self.busy = False


def html_escape(text):
    import html
    return html.escape(text)


class TemplateTracker:
    """Core-OpenCV normalized-correlation tracker; no contrib package required.

    Tracks an appearance template with a small amount of surrounding context.
    A low match score reports loss rather than inventing a new location.
    """

    def __init__(self, min_score=0.35):
        self.min_score = float(min_score)
        self.template = None
        self.bbox = None

    def init(self, frame, bbox):
        height, width = frame.shape[:2]
        x, y, w, h = [int(round(v)) for v in bbox]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(width, x + w), min(height, y + h)
        if x2 - x1 < 3 or y2 - y1 < 3:
            raise ValueError("Template tracker received an invalid initial box")
        self.bbox = (x1, y1, x2 - x1, y2 - y1)
        margin = max(8, int(0.15 * max(w, h)))
        tx1, ty1 = max(0, x1 - margin), max(0, y1 - margin)
        tx2, ty2 = min(width, x2 + margin), min(height, y2 + margin)
        self.offset = (x1 - tx1, y1 - ty1)
        self.template = frame[ty1:ty2, tx1:tx2].copy()
        return True

    def update(self, frame):
        if self.template is None or self.bbox is None:
            return False, self.bbox
        height, width = frame.shape[:2]
        x, y, w, h = self.bbox
        expected_x, expected_y = x - self.offset[0], y - self.offset[1]
        radius = max(35, int(0.8 * max(w, h)))
        th, tw = self.template.shape[:2]
        sx1, sy1 = max(0, expected_x - radius), max(0, expected_y - radius)
        sx2, sy2 = min(width, expected_x + tw + radius), min(height, expected_y + th + radius)
        search = frame[sy1:sy2, sx1:sx2]
        if search.shape[0] < th or search.shape[1] < tw or search.size == 0:
            return False, tuple(float(v) for v in self.bbox)
        try:
            response = cv2.matchTemplate(search, self.template, cv2.TM_CCOEFF_NORMED)
            _, score, _, location = cv2.minMaxLoc(response)
        except cv2.error:
            return False, tuple(float(v) for v in self.bbox)
        if not np.isfinite(score) or score < self.min_score:
            return False, tuple(float(v) for v in self.bbox)
        template_x, template_y = sx1 + location[0], sy1 + location[1]
        new_x, new_y = template_x + self.offset[0], template_y + self.offset[1]
        self.bbox = (int(new_x), int(new_y), int(w), int(h))
        if score >= 0.55:
            current = search[location[1]:location[1] + th, location[0]:location[0] + tw]
            if current.shape == self.template.shape:
                self.template = cv2.addWeighted(self.template, 0.95, current, 0.05, 0)
        return True, tuple(float(v) for v in self.bbox)


def create_tracker(name):
    name = name.lower()
    if name in ("template", "template_ncc"):
        return TemplateTracker(), "template_ncc"
    constructors = {
        "csrt": ["TrackerCSRT_create", "legacy.TrackerCSRT_create"],
        "kcf": ["TrackerKCF_create", "legacy.TrackerKCF_create"],
        "mosse": ["legacy.TrackerMOSSE_create", "TrackerMOSSE_create"],
    }
    for attribute in constructors[name]:
        owner = cv2
        try:
            for part in attribute.split("."):
                owner = getattr(owner, part)
            return owner(), name
        except (AttributeError, cv2.error):
            continue
    if name != "kcf":
        print(f"Requested {name} tracker unavailable; trying KCF before the core fallback.")
        return create_tracker("kcf")
    print("OpenCV contrib trackers unavailable; using core template_ncc and recording it.")
    return TemplateTracker(), "template_ncc"


def color_for(label, index):
    return COLORS.get(label.lower(), FALLBACK_COLORS[index % len(FALLBACK_COLORS)])


def valid_box(box, width, height):
    if box is None or len(box) != 4:
        return False
    x, y, w, h = [float(v) for v in box]
    return w >= 3 and h >= 3 and w < width * 1.5 and h < height * 1.5 and x < width and y < height and x + w > 0 and y + h > 0


def clipped_xyxy(box, width, height):
    x, y, w, h = [float(v) for v in box]
    x1, y1 = max(0, min(width - 1, x)), max(0, min(height - 1, y))
    x2, y2 = max(0, min(width - 1, x + w)), max(0, min(height - 1, y + h))
    return [x1, y1, x2, y2]


DETECTION_CONFIG = {
    "processing_width": 960,
    "hsv_v_min": 170,
    "hsv_s_max": 90,
    "open_kernel": 3,
    "close_kernel": 3,
    "green_fraction_max": 0.20,
    "min_center_y_fraction": 0.30,
    "min_contour_area": 500,
    "max_contour_area": 30000,
    "min_aspect": 0.42,
    "max_aspect": 1.70,
    "min_extent": 0.40,
    "min_solidity": 0.68,
    "min_circularity": 0.30,
    "min_area_ratio": 0.30,
    "max_area_ratio": 4.0,
    "reacquire_after_misses": 6,
}


def detect_egg_candidates(frame, config=DETECTION_CONFIG):
    """Return conservative light-egg candidates and whether the scene is eligible.

    The green-card gate is important for this supplied video: white explanation
    text and logos must not become eggs after the table sequence ends.
    """
    height, width = frame.shape[:2]
    scale = min(1.0, float(config["processing_width"]) / width)
    if scale < 1.0:
        small = cv2.resize(frame, (int(round(width * scale)), int(round(height * scale))),
                           interpolation=cv2.INTER_AREA)
    else:
        small = frame.copy()
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([35, 60, 40], np.uint8), np.array([95, 255, 255], np.uint8))
    if float(np.mean(green > 0)) > config["green_fraction_max"]:
        return [], False
    mask = cv2.inRange(hsv, np.array([0, 0, config["hsv_v_min"]], np.uint8),
                       np.array([179, config["hsv_s_max"], 255], np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            np.ones((config["open_kernel"], config["open_kernel"]), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            np.ones((config["close_kernel"], config["close_kernel"]), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not config["min_contour_area"] <= area <= config["max_contour_area"]:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 18 or h < 18:
            continue
        aspect = w / float(h)
        extent = area / float(w * h)
        if not config["min_aspect"] <= aspect <= config["max_aspect"] or extent < config["min_extent"]:
            continue
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / hull_area if hull_area else 0.0
        perimeter = float(cv2.arcLength(contour, True))
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter else 0.0
        if solidity < config["min_solidity"] or circularity < config["min_circularity"]:
            continue
        box = np.array([x / scale, y / scale, (x + w) / scale, (y + h) / scale], dtype=float)
        center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=float)
        if center[1] < config["min_center_y_fraction"] * height:
            continue
        candidates.append({
            "box": box, "center": center, "area": area / (scale * scale), "aspect": aspect,
            "extent": extent, "solidity": solidity, "circularity": circularity,
            "shape_score": min(1.0, (extent + solidity + min(1.0, circularity)) / 3),
        })
    return candidates, True


class DetectionTrackState:
    def __init__(self, label, box):
        self.label = label
        self.box = np.asarray(box, dtype=float)
        self.center = np.array([(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2])
        self.velocity = np.zeros(2, dtype=float)
        area = max(1.0, (self.box[2] - self.box[0]) * (self.box[3] - self.box[1]))
        aspect = (self.box[2] - self.box[0]) / max(1.0, self.box[3] - self.box[1])
        # Slow, gated adaptation prevents tiny hand/nail candidates from shrinking
        # the object's size reference and then becoming self-sustaining false tracks.
        self.areas = deque([area], maxlen=15)
        self.aspects = deque([aspect], maxlen=15)
        self.missed = 0
        self.miss_streak = 0
        self.max_miss = 0

    def mark_missed(self):
        self.velocity *= 0.7
        self.missed += 1
        self.miss_streak += 1
        self.max_miss = max(self.max_miss, self.miss_streak)

    def update(self, detection):
        old_center = self.center.copy()
        instant_velocity = detection["center"] - old_center
        self.velocity = 0.7 * self.velocity + 0.3 * instant_velocity
        self.box = detection["box"]
        self.center = detection["center"]
        area_ref = float(np.median(self.areas))
        area_ratio = detection["area"] / area_ref
        if 0.45 <= area_ratio <= 2.2 and detection["shape_score"] >= 0.60:
            self.areas.append(detection["area"])
            self.aspects.append(detection["aspect"])
        self.missed = 0
        self.miss_streak = 0


def associate_detections(tracks, detections, width, height, config=DETECTION_CONFIG):
    """Greedy one-to-one association; every candidate can belong to only one ID."""
    pairs = []
    for track_index, track in enumerate(tracks):
        predicted = track.center + (track.velocity if track.missed < 5 else 0)
        diagonal = float(np.linalg.norm(track.box[2:] - track.box[:2]))
        gate = max(0.073 * width, 1.25 * diagonal)
        if track.missed > config["reacquire_after_misses"]:
            gate = math.hypot(width, height)
        area_ref = float(np.median(track.areas))
        aspect_ref = float(np.median(track.aspects))
        for detection_index, detection in enumerate(detections):
            area_ratio = detection["area"] / area_ref
            if not config["min_area_ratio"] <= area_ratio <= config["max_area_ratio"]:
                continue
            distance = float(np.linalg.norm(detection["center"] - predicted))
            if distance > gate:
                continue
            size_cost = abs(math.log(max(area_ratio, 1e-6)))
            aspect_cost = abs(math.log(max(detection["aspect"] / aspect_ref, 1e-6)))
            cost = (distance / gate + 0.45 * size_cost + 0.15 * aspect_cost
                    + 0.15 * (1 - detection["shape_score"]))
            pairs.append((cost, track_index, detection_index))
    pairs.sort(key=lambda item: item[0])
    assigned_tracks, assigned_detections, mapping = set(), set(), {}
    for _, track_index, detection_index in pairs:
        if track_index in assigned_tracks or detection_index in assigned_detections:
            continue
        assigned_tracks.add(track_index)
        assigned_detections.add(detection_index)
        mapping[track_index] = detection_index
    return mapping


def track_detection(video_path, scan, annotations, output_video, tracks_csv):
    """Detection-gated full-video tracking with explicit no-detection states."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Cannot open source video for detection tracking.")
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), scan["fps"],
                             (scan["width"], scan["height"]))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Cannot create output MP4.")
    rows, total_processed = [], 0
    assignment_counts = {}
    try:
        for segment in scan["segments"]:
            sid = segment["segment_id"]
            cap.set(cv2.CAP_PROP_POS_FRAMES, segment["start_frame"])
            tracks = []
            for frame_index in range(segment["start_frame"], segment["end_frame"]):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Video ended unexpectedly at frame {frame_index}")
                first_frame = frame_index == segment["start_frame"]
                if first_frame:
                    for item in annotations["annotations"][sid]:
                        tracks.append(DetectionTrackState(item["object_id"], item["box_xyxy"]))
                detections, active_scene = detect_egg_candidates(frame) if tracks else ([], False)
                mapping = associate_detections(tracks, detections, scan["width"], scan["height"]) if active_scene else {}
                if tracks and active_scene:
                    key = str(len(mapping))
                    assignment_counts[key] = assignment_counts.get(key, 0) + 1
                for label_index, track in enumerate(tracks):
                    detection = detections[mapping[label_index]] if label_index in mapping else None
                    if detection is not None:
                        track.update(detection)
                        status = "tracked"
                        box = track.box
                        confidence = round(float(detection["shape_score"]), 4)
                    elif first_frame:
                        # The human explicitly confirmed this box at the segment start.
                        status = "initialized"
                        box = track.box
                        confidence = ""
                    else:
                        track.mark_missed()
                        status = "out_of_scene" if not active_scene else "not_detected"
                        box = None
                        confidence = ""
                    if box is None:
                        x1 = y1 = x2 = y2 = ""
                    else:
                        x1, y1, x2, y2 = [float(v) for v in box]
                    rows.append({"frame": frame_index, "seconds": frame_index / scan["fps"], "segment_id": sid,
                                 "object_id": track.label, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                 "status": status, "confidence": confidence})
                    color = color_for(track.label, label_index)
                    if box is not None:
                        bx1, by1, bx2, by2 = [int(round(v)) for v in box]
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 3)
                        cv2.putText(frame, track.label, (bx1, max(28, by1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
                    elif status == "not_detected":
                        cv2.putText(frame, f"{track.label}: not detected", (20, 38 + 34 * label_index),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                cv2.putText(frame, f"frame {frame_index} | {sid}", (20, scan["height"] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                writer.write(frame)
                total_processed += 1
                if total_processed % 500 == 0:
                    print(f"Detection-tracked/written {total_processed}/{scan['frames']} frames", flush=True)
    finally:
        cap.release()
        writer.release()
    if total_processed != scan["frames"]:
        raise RuntimeError(f"Only {total_processed}/{scan['frames']} frames processed; full-video output is not accepted.")
    with tracks_csv.open("w", newline="", encoding="utf-8") as handle:
        fields = ["frame", "seconds", "segment_id", "object_id", "x1", "y1", "x2", "y2", "status", "confidence"]
        writer_csv = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    object_stats = {}
    for row in rows:
        stats = object_stats.setdefault(row["object_id"], {"tracked": 0, "initialized": 0,
                                                           "not_detected": 0, "out_of_scene": 0,
                                                           "first_visible_frame": None, "last_visible_frame": None})
        stats[row["status"]] += 1
        if row["status"] in ("tracked", "initialized"):
            stats["first_visible_frame"] = row["frame"] if stats["first_visible_frame"] is None else stats["first_visible_frame"]
            stats["last_visible_frame"] = row["frame"]
    return {"frames_processed": total_processed, "track_rows": len(rows),
            "lost_object_frames": sum(r["status"] == "not_detected" for r in rows),
            "out_of_scene_object_frames": sum(r["status"] == "out_of_scene" for r in rows),
            "actual_tracker": "detection_gated_hsv", "object_ids": sorted(object_stats),
            "assignment_counts": assignment_counts, "object_stats": object_stats,
            "detector_config": DETECTION_CONFIG}


def track(video_path, scan, annotations, output_video, tracks_csv, tracker_name):
    cap = cv2.VideoCapture(str(video_path))
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), scan["fps"],
                             (scan["width"], scan["height"]))
    if not writer.isOpened():
        raise RuntimeError("Cannot create output MP4.")
    actual_tracker = tracker_name
    rows, total_processed, lost_frames = [], 0, 0
    try:
        for segment in scan["segments"]:
            sid = segment["segment_id"]
            cap.set(cv2.CAP_PROP_POS_FRAMES, segment["start_frame"])
            trackers, labels, last_boxes = {}, [], {}
            for frame_index in range(segment["start_frame"], segment["end_frame"]):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Video ended unexpectedly at frame {frame_index}")
                if frame_index == segment["start_frame"]:
                    for item in annotations["annotations"][sid]:
                        label = item["object_id"]
                        x1, y1, x2, y2 = item["box_xyxy"]
                        tracker, actual_tracker = create_tracker(tracker_name)
                        init_box = tuple(int(round(v)) for v in (x1, y1, x2 - x1, y2 - y1))
                        tracker.init(frame, init_box)
                        trackers[label] = tracker
                        labels.append(label)
                        last_boxes[label] = tuple(float(v) for v in init_box)
                        rows.append({"frame": frame_index, "seconds": frame_index / scan["fps"], "segment_id": sid,
                                     "object_id": label, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                     "status": "initialized", "confidence": ""})
                else:
                    for label in labels:
                        ok_track, box = trackers[label].update(frame)
                        if ok_track and valid_box(box, scan["width"], scan["height"]):
                            last_boxes[label] = box
                            x1, y1, x2, y2 = clipped_xyxy(box, scan["width"], scan["height"])
                            status = "tracked"
                        else:
                            x1 = y1 = x2 = y2 = ""
                            status = "lost"
                            lost_frames += 1
                        rows.append({"frame": frame_index, "seconds": frame_index / scan["fps"], "segment_id": sid,
                                     "object_id": label, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                     "status": status, "confidence": ""})
                for label_index, label in enumerate(labels):
                    color = color_for(label, label_index)
                    row = next(r for r in reversed(rows) if r["frame"] == frame_index and r["object_id"] == label)
                    if row["status"] != "lost":
                        x1, y1, x2, y2 = [int(float(row[k])) for k in ("x1", "y1", "x2", "y2")]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                        cv2.putText(frame, label, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    else:
                        cv2.putText(frame, f"{label}: LOST", (20, 35 + 30 * label_index),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                cv2.putText(frame, f"frame {frame_index} | {sid}", (20, scan["height"] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                writer.write(frame)
                total_processed += 1
                if total_processed % 500 == 0:
                    print(f"Tracked/written {total_processed}/{scan['frames']} frames", flush=True)
    finally:
        cap.release()
        writer.release()
    if total_processed != scan["frames"]:
        raise RuntimeError(f"Only {total_processed}/{scan['frames']} frames processed; full-video output is not accepted.")
    with tracks_csv.open("w", newline="", encoding="utf-8") as handle:
        fields = ["frame", "seconds", "segment_id", "object_id", "x1", "y1", "x2", "y2", "status", "confidence"]
        writer_csv = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    return {"frames_processed": total_processed, "track_rows": len(rows), "lost_object_frames": lost_frames,
            "actual_tracker": actual_tracker, "object_ids": sorted({r["object_id"] for r in rows})}


def main(project_root, video=None, tracker="detection", redo=False):
    project = Path(project_root)
    video_path = Path(video) if video else project / "data" / "video" / "input.mp4"
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}. Upload the authorized full video, then rerun.")
    results, work = project / "results", project / "video_work"
    results.mkdir(exist_ok=True)
    work.mkdir(exist_ok=True)
    summary_path = results / "video_tracking_summary.json"
    if summary_path.exists() and not redo:
        print("Video tracking already complete. Use --redo only for a deliberate, documented correction.")
        print(summary_path)
        return json.loads(summary_path.read_text())
    if redo and summary_path.exists():
        archive = results / f"video_archive_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        archive.mkdir()
        for name in ("video_tracking_summary.json", "video_tracks.csv", "video_tracked.mp4"):
            path = results / name
            if path.exists():
                shutil.move(str(path), str(archive / name))
    video_hash = file_sha256(video_path)
    scan_path = work / "scene_scan.json"
    if scan_path.exists():
        scan = json.loads(scan_path.read_text())
        if scan["video_sha256"] != video_hash:
            raise RuntimeError("Video changed but old scene scan exists. Use a new video_work folder; do not mix runs.")
    else:
        print("Pass 1/2: scanning the full video for candidate cuts...")
        scan = scan_video(video_path)
        scan["video_sha256"] = video_hash
        scan["source_video"] = video_path.name
        scan["scan_sha256"] = file_sha256(video_path) + ":" + json.dumps(scan["detector"], sort_keys=True)
        scan_path.write_text(json.dumps(scan, indent=2) + "\n")
    print(f"Video: {scan['frames']} frames | {scan['fps']:.3f} fps | {scan['width']}x{scan['height']} | {len(scan['segments'])} segments")
    def finish_tracking(annotations):
        print("Pass 2/2: tracking and writing every frame...")
        if tracker == "detection":
            stats = track_detection(video_path, scan, annotations, results / "video_tracked.mp4", results / "video_tracks.csv")
        else:
            stats = track(video_path, scan, annotations, results / "video_tracked.mp4", results / "video_tracks.csv", tracker)
        summary = {
            "status": "FULL_VIDEO_TRACKED", "created_at": now(), "source_video_name": video_path.name,
            "source_video_sha256": video_hash, **{k: scan[k] for k in ("fps", "width", "height", "frames")},
            "candidate_segments": len(scan["segments"]), "segments": scan["segments"],
            "cut_detection": scan["detector"], **stats,
            "identity_policy": "User assigns persistent labels at each candidate segment start; one-to-one temporal association preserves those IDs. No automatic raw/boiled recognition is claimed.",
            "loss_policy": "not_detected and out_of_scene rows have blank coordinates; positions are not interpolated or fabricated. Green cards suppress boxes and status overlays.",
            "audio": "OpenCV visualization contains no audio.",
            "limitations": "Candidate cut detection may oversegment/undersegment. Detection mode is calibrated for light eggs on the supplied table/green-card video; hand occlusion can cause missed detections. Review the corrected playback for identity switches and false boxes before submission.",
            "ai_assistance": "Tracking code developed with Arena.ai; user supplies boxes, executes and reviews the full result.",
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print("\nFULL VIDEO TRACKING COMPLETE")
        print(json.dumps(stats, indent=2))
        print("Review video_tracked.mp4, then upload it to Drive and set link access to anyone with the link (viewer).")
        return summary

    global VIDEO_REVIEW_UI
    VIDEO_REVIEW_UI = VideoAnnotationUI(video_path, scan, work, finish_tracking)
    VIDEO_REVIEW_UI.start()
    return VIDEO_REVIEW_UI


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default="/content/drive/MyDrive/Egg_CV_Task")
    parser.add_argument("--video", default=None)
    parser.add_argument("--tracker", choices=["detection", "csrt", "kcf", "mosse", "template"], default="detection")
    parser.add_argument("--redo", action="store_true")
    args = parser.parse_args()
    main(args.project_root, args.video, args.tracker, args.redo)
