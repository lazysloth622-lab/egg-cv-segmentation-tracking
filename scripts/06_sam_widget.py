"""Button/widget-based SAM reference reviewer for Colab (no input() calls).

Use this when Colab does not render the text prompt from 06_sam_reference_masks.py:
  %run "/content/drive/MyDrive/Egg_CV_Task/scripts/06_sam_widget.py"

Outputs are identical to Step 06 and resume from already accepted annotations.
Select candidate numbers in the visible widget, or use the quick-number buttons.
For multiple eggs enter 0,3. For manual boxes enter x1,y1,x2,y2; ... and click
USE BOXES. Review the final overlay, then click ACCEPT & NEXT.

GPU quota fallback (slower, manual boxes only):
  %run "/content/drive/MyDrive/Egg_CV_Task/scripts/06_sam_widget.py" --cpu --boxes-only

No training occurs. Developed with Arena.ai assistance; human review is required.
"""

from pathlib import Path
import argparse
import importlib.util
import json
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import ipywidgets as widgets
from IPython.display import display, clear_output

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Reuse the tested mask/prompt/save helpers without running the input() workflow.
spec = importlib.util.spec_from_file_location("sam_reference_helpers", SCRIPT_DIR / "06_sam_reference_masks.py")
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


class SamReviewUI:
    def __init__(self, project_root, only="", redo=False, max_images=None, cpu=False, boxes_only=False):
        self.project = Path(project_root)
        self.device = "cpu" if cpu else "cuda"
        self.boxes_only = boxes_only
        self.metadata = self.project / "metadata"
        self.reference = self.project / "references"
        self.reference.mkdir(exist_ok=True)
        self.busy = False
        self.finished = False
        self.empty_armed = False
        self.instances = []
        self.prompts = []
        self.index = -1
        split_path = self.metadata / "split.csv"
        freeze = json.loads((self.metadata / "split_freeze.json").read_text())
        if h.file_sha256(split_path) != freeze["file_sha256"]["split.csv"]:
            raise RuntimeError("Frozen split changed.")
        environment = json.loads((self.metadata / "sam_environment.json").read_text())
        if environment["split_sha256"] != h.file_sha256(split_path):
            raise RuntimeError("SAM environment belongs to another split.")
        self.environment = environment
        self.split_path = split_path
        self.all_rows = h.read_csv(split_path)
        rows = list(self.all_rows)
        selected = {x.strip().upper() for x in only.split(",") if x.strip()}
        if selected:
            unknown = selected - {r["image_id"] for r in rows}
            if unknown:
                raise ValueError(f"Unknown IDs: {sorted(unknown)}")
            rows = [r for r in rows if r["image_id"] in selected]
        if redo and not selected:
            raise ValueError("--redo requires --only I001,I002,...")
        if max_images:
            rows = rows[:max_images]
        self.selected_count = len(rows)
        self.rows = []
        for row in rows:
            annotation = self.reference / "annotations" / f"{row['image_id']}.json"
            if annotation.exists() and not redo:
                continue
            self.rows.append(row)
        h.compile_metadata(self.project, self.all_rows)
        if not self.rows:
            print("Nothing pending. All selected images are already reviewed.")
            self.finished = True
            return
        if not cpu and not torch.cuda.is_available():
            raise RuntimeError("GPU is unavailable. Enable T4, or use --cpu --boxes-only for the slower manual-box fallback.")
        from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
        checkpoint = self.project / environment["checkpoint_relative_path"]
        if h.file_sha256(checkpoint) != environment["checkpoint"]["sha256"]:
            raise RuntimeError("Checkpoint hash mismatch.")
        self._ensure_policy()
        print(f"Loading pretrained SAM on {self.device}. No training will occur.")
        if cpu:
            torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        model = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
        model.to(self.device)
        model.eval()
        self.predictor = SamPredictor(model)
        self.generator = None if boxes_only else SamAutomaticMaskGenerator(
            model=model, output_mode="binary_mask", **h.GENERATOR_CONFIG
        )
        self._build_widgets()

    def _ensure_policy(self):
        policy_path = self.reference / "reference_policy.json"
        policy = {
            "policy_version": "visible_egg_pixels_only_v1", "status": "HUMAN_REVIEWED_SILVER_NOT_INDEPENDENT_GOLD",
            "model_type": "vit_b", "sam_commit": self.environment["sam_commit"],
            "checkpoint_sha256": self.environment["checkpoint"]["sha256"], "generator": h.GENERATOR_CONFIG,
            "split_sha256": h.file_sha256(self.split_path),
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

    def _build_widgets(self):
        self.title = widgets.HTML()
        self.image_out = widgets.Output()
        self.log_out = widgets.Output(layout={"height": "90px", "overflow_y": "auto"})
        self.choice = widgets.Text(description="Candidates", placeholder="0 or 0,3", layout=widgets.Layout(width="260px"))
        self.use_candidates = widgets.Button(description="USE CANDIDATES", button_style="primary")
        self.boxes = widgets.Text(description="Boxes", placeholder="x1,y1,x2,y2; ...", layout=widgets.Layout(width="430px"))
        self.use_boxes = widgets.Button(description="USE BOXES")
        self.accept = widgets.Button(description="ACCEPT & NEXT", button_style="success")
        self.retry = widgets.Button(description="RETRY")
        self.skip = widgets.Button(description="SKIP")
        self.empty = widgets.Button(description="NO VISIBLE EGG")
        self.quit = widgets.Button(description="SAVE & STOP", button_style="danger")
        self.quick = [widgets.Button(description=str(i), layout=widgets.Layout(width="42px")) for i in range(h.MAX_CANDIDATES)]
        self.quick_row = widgets.HBox(self.quick)
        self.candidate_row = widgets.HBox([self.choice, self.use_candidates])
        self.box_row = widgets.HBox([self.boxes, self.use_boxes])
        self.review_row = widgets.HBox([self.accept, self.retry])
        self.nav_row = widgets.HBox([self.skip, self.empty, self.quit])
        self.root = widgets.VBox([
            self.title, self.image_out, self.quick_row, self.candidate_row,
            self.box_row, self.review_row, self.nav_row,
            widgets.HTML("<b>Quick buttons select one candidate. For several eggs use Candidates = 0,3. Boxes use ORIGINAL axes.</b>"),
            self.log_out,
        ])
        self.use_candidates.on_click(lambda _: self.choose_candidates(self.choice.value))
        self.use_boxes.on_click(lambda _: self.choose_boxes(self.boxes.value))
        self.accept.on_click(lambda _: self.accept_current())
        self.retry.on_click(lambda _: self.show_candidates())
        self.skip.on_click(lambda _: self.advance())
        self.empty.on_click(lambda _: self.mark_empty())
        self.quit.on_click(lambda _: self.stop())
        for i, button in enumerate(self.quick):
            button.on_click(lambda _, index=i: self.choose_candidates(str(index)))
        self._set_mode("loading")

    def _set_mode(self, mode):
        candidate_visible = mode == "candidates"
        box_visible = candidate_visible or mode == "boxes"
        review_visible = mode == "review"
        for widget in [self.quick_row, self.candidate_row]:
            widget.layout.display = "flex" if candidate_visible else "none"
        self.box_row.layout.display = "flex" if box_visible else "none"
        self.review_row.layout.display = "flex" if review_visible else "none"
        self.nav_row.layout.display = "none" if self.finished else "flex"

    def log(self, message):
        with self.log_out:
            print(message)

    def start(self):
        if self.finished:
            return
        display(self.root)
        self.advance()

    @property
    def current(self):
        return self.rows[self.index]

    def advance(self):
        if self.busy or self.finished:
            return
        self.index += 1
        self.empty_armed = False
        self.empty.description = "NO VISIBLE EGG"
        if self.index >= len(self.rows):
            self.finish()
            return
        self.busy = True
        self._set_mode("loading")
        row = self.current
        action = "Loading original for manual box review…" if self.boxes_only else "Generating SAM candidates…"
        self.title.value = f"<h3>[{self.index+1}/{len(self.rows)}] {row['image_id']} | {row['split']}</h3><p>{action}</p>"
        try:
            path = self.project / "data" / "raw" / row["relative_path"]
            if h.file_sha256(path) != row["sha256"]:
                raise RuntimeError("Raw image hash mismatch")
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise RuntimeError("Unreadable image")
            self.rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if self.boxes_only:
                self.candidates = []
            else:
                with torch.inference_mode():
                    generated = self.generator.generate(self.rgb)
                ranked = [h.candidate_rank(item, self.rgb.shape[:2]) for item in generated]
                candidates = [x for x in ranked if x is not None]
                candidates.sort(key=lambda x: x["review_rank_score"], reverse=True)
                self.candidates = h.deduplicate(candidates)[:h.MAX_CANDIDATES]
            self.choice.value = ""
            self.boxes.value = ""
            self.show_candidates()
            if self.boxes_only:
                self.log(f"{row['image_id']}: {self.device} box-only mode. Enter one tight box per visible egg.")
            else:
                self.log(f"{row['image_id']}: {len(self.candidates)} candidates. Select every visible egg.")
        except Exception as exc:
            self.title.value = f"<h3>{row['image_id']} — generation error</h3>"
            self.log(f"ERROR: {exc}. Use SKIP or SAVE & STOP; share the error.")
            with self.image_out:
                clear_output(wait=True)
                print(exc)
            self._set_mode("error")
        finally:
            self.busy = False

    def show_candidates(self):
        if self.finished:
            return
        row = self.current
        if self.boxes_only:
            self.title.value = (f"<h3>[{self.index+1}/{len(self.rows)}] {row['image_id']} | {row['split']} | {self.device} box-only</h3>"
                                "<p>Use the axes: enter one tight box per visible egg, separated by semicolons, then click USE BOXES.</p>")
            with self.image_out:
                clear_output(wait=True)
                plt.close("all")
                fig, axis = plt.subplots(figsize=(8, 8))
                axis.imshow(self.rgb)
                axis.set_title(f"{row['image_id']} ORIGINAL — enter egg boxes below")
                axis.grid(alpha=0.3)
                display(fig)
                plt.close(fig)
            self._set_mode("boxes")
            return
        self.title.value = (f"<h3>[{self.index+1}/{len(self.rows)}] {row['image_id']} | {row['split']}</h3>"
                            "<p>Click a quick number OR enter 0,3 and USE CANDIDATES. Manual boxes are also available.</p>")
        for i, button in enumerate(self.quick):
            button.layout.display = "inline-block" if i < len(self.candidates) else "none"
        with self.image_out:
            clear_output(wait=True)
            h.show_candidates(self.rgb, self.candidates, row["image_id"], row["split"])
        self._set_mode("candidates")

    def _refine(self, results):
        self.instances = [(mask > 0).astype(np.uint8) for mask, _ in results]
        self.prompts = [prompt for _, prompt in results]
        with self.image_out:
            clear_output(wait=True)
            h.show_final(self.rgb, self.instances, self.current["image_id"])
        self.title.value = f"<h3>{self.current['image_id']}: REVIEW FINAL MASK</h3><p>If every visible egg is correct, click ACCEPT & NEXT. Otherwise RETRY.</p>"
        self._set_mode("review")

    def choose_candidates(self, text):
        if self.busy or self.finished:
            return
        try:
            indices = [int(x.strip()) for x in str(text).split(",")]
            if not indices or len(set(indices)) != len(indices) or any(i < 0 or i >= len(self.candidates) for i in indices):
                raise ValueError("Enter unique visible candidate numbers, e.g. 0 or 0,3")
            self.busy = True
            self.title.value = f"<h3>{self.current['image_id']}: refining selected egg prompts…</h3>"
            with torch.inference_mode():
                self.predictor.set_image(self.rgb)
                results = [h.prompt_from_candidate(self.predictor, self.candidates[i]) for i in indices]
            for _, prompt in results:
                prompt["runtime_device"] = self.device
            self._refine(results)
        except Exception as exc:
            self.log(f"Selection error: {exc}")
        finally:
            self.busy = False

    def choose_boxes(self, text):
        if self.busy or self.finished:
            return
        try:
            boxes = h.parse_boxes(text, self.rgb.shape[1], self.rgb.shape[0])
            self.busy = True
            self.title.value = f"<h3>{self.current['image_id']}: applying human box prompts…</h3>"
            with torch.inference_mode():
                self.predictor.set_image(self.rgb)
                results = [h.prompt_from_box(self.predictor, box) for box in boxes]
            for _, prompt in results:
                prompt["runtime_device"] = self.device
            self._refine(results)
        except Exception as exc:
            self.log(f"Box error: {exc}. Format: x1,y1,x2,y2; x1,y1,x2,y2")
        finally:
            self.busy = False

    def accept_current(self):
        if self.busy or self.finished or not self.instances:
            return
        try:
            h.save_annotation(self.project, self.current, self.rgb, self.instances, self.prompts)
            h.compile_metadata(self.project, self.all_rows)
            self.log(f"SAVED {self.current['image_id']} with {len(self.instances)} instance(s).")
            self.instances, self.prompts = [], []
            self.advance()
        except Exception as exc:
            self.log(f"Save error: {exc}")

    def mark_empty(self):
        if self.busy or self.finished:
            return
        if not self.empty_armed:
            self.empty_armed = True
            self.empty.description = "CONFIRM EMPTY"
            self.log("Click CONFIRM EMPTY only if there is genuinely no visible egg.")
            return
        h.save_annotation(self.project, self.current, self.rgb, [], [], status="reviewed_empty")
        h.compile_metadata(self.project, self.all_rows)
        self.log(f"SAVED {self.current['image_id']} as reviewed empty.")
        self.advance()

    def stop(self):
        h.compile_metadata(self.project, self.all_rows)
        self.finished = True
        self._set_mode("stopped")
        self.title.value = "<h3>Stopped safely</h3><p>Accepted images are saved. Rerun this widget to resume.</p>"

    def finish(self):
        _, status = h.compile_metadata(self.project, self.all_rows)
        reviewed = sum(r["status"].startswith("reviewed") for r in status)
        self.finished = True
        self._set_mode("finished")
        self.title.value = f"<h3>Reviewed {reviewed}/110 | Pending {110-reviewed}</h3>"
        with self.image_out:
            clear_output(wait=True)
            if reviewed == 110:
                print("ALL REFERENCES COMPLETE. Next: 07_classical_dev_experiments.py")
            else:
                print("This selected batch is complete. Rerun for remaining pending images.")


def main(project_root, only="", redo=False, max_images=None, cpu=False, boxes_only=False):
    global SAM_REVIEW_UI
    SAM_REVIEW_UI = SamReviewUI(project_root, only, redo, max_images, cpu, boxes_only)
    if not SAM_REVIEW_UI.finished:
        SAM_REVIEW_UI.start()
    return SAM_REVIEW_UI


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default="/content/drive/MyDrive/Egg_CV_Task")
    parser.add_argument("--only", default="")
    parser.add_argument("--redo", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--cpu", action="store_true", help="Run SAM on CPU; use with --boxes-only when GPU quota is unavailable")
    parser.add_argument("--boxes-only", action="store_true", help="Skip expensive automatic candidates and use manual egg boxes")
    args = parser.parse_args()
    main(args.project_root, args.only, args.redo, args.max_images, args.cpu, args.boxes_only)
