# Egg Segmentation and Tracking — Execution Report

**Status:** COMPLETE_LOCAL_RESULTS  
**Generated UTC:** 2026-10-31T13:51:32.174068+00:00  
**AI assistance:** Planning, draft visual grouping, code and documentation were assisted by Arena.ai. The user executed the pipeline and was responsible for prompt selection, mask review and final verification. No result in this report is intended to represent unexecuted work.

## 1. Task and dataset

- 110 original 640×640 JPG images from the supplied Roboflow v1 classification dataset; original train/valid/test labels were retained only as source metadata.
- The task uses pretrained SAM for reference-mask creation, a separate classical segmentation method, group-aware development/evaluation, and individual egg tracking on the full supplied video.
- The source dataset has no independent pixel-level gold masks. References here are **human-reviewed SAM-derived silver masks**, so classical scores measure agreement with those references and are not absolute accuracy against independent human ground truth.

## 2. Leakage-reduction split

- Visual scene families were grouped before segmentation experiments; 88 development and 22 held-out evaluation images were frozen with seed 42.
- Declared scene groups crossing the split: **0**.
- Evaluation groups: **G06, G07, G08, G09, G10, G12, G13**.
- Conservative kitchen family G11 has 58 images, including the clearly visible multi-egg scenes, and remains development-only. Therefore held-out multi-egg and scene diversity are limited. This is reported rather than breaking related scenes to force coverage.
- Visual families reduce obvious setup leakage but cannot prove unique people, physical eggs or capture sessions without provenance.

## 3. SAM reference creation

- Model: **pretrained SAM ViT-B**; no SAM training or fine-tuning was performed.
- Official repository commit: `dca509fe793f601edb92606367a655c15ac00fdf`.
- Checkpoint SHA-256: `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`.
- Reviewed references: **110/110**.
- Automatic SAM grid candidates were used for prompt discovery. Accepted instances record per-egg point+box or human-box prompts and a final human review decision.
- Label policy: visible egg pixels only; hands/background excluded; hidden egg regions were not completed. Binary union and per-instance PNGs, prompts, overlays and review status are saved.

## 4. Classical method and DEV-only experiments

Classical inference receives only the original image and a frozen configuration. It does **not** receive SAM prompts, reference boxes or reference masks. Operations include HSV/Lab appearance seeds, morphology, connected-component filtering and/or GrabCut depending on the selected configuration.

- Fixed configurations tried on development images: **16**.
- Selection rule: maximum macro development mean IoU; ties broken by mean Dice then micro IoU.
- Selected configuration: **C08_grabcut_center**.
- Evaluation references and metrics were not used for selection. The final algorithm was frozen before evaluation reference pixels were read.

```json
{
  "center_patch": 0.16,
  "center_weight": 1.0,
  "close_kernel": 5,
  "component_filter": true,
  "grabcut_iters": 4,
  "lab_scales": [
    45.0,
    22.0,
    22.0
  ],
  "lab_threshold": 1.2,
  "max_area_frac": 0.65,
  "max_components": 2,
  "max_side": 384,
  "method": "grabcut_rect",
  "min_area_frac": 0.001,
  "open_kernel": 3,
  "rect_scale": 0.5,
  "sat_max": 80,
  "sure_bg_kernel": 15,
  "sure_fg_kernel": 9,
  "val_min": 70,
  "warm": false,
  "warm_hue_max": 35,
  "warm_hue_min": 3,
  "warm_sat_max": 180
}
```

### Development trial summary

| Config | Method | IoU | Dice | Precision | Recall | Errors |
|---|---|---|---|---|---|---|
| C08_grabcut_center | grabcut_rect | 0.5107 | 0.5975 | 0.6367 | 0.6182 | 0 |
| C09_grabcut_wide | grabcut_rect | 0.4950 | 0.5976 | 0.5217 | 0.7990 | 0 |
| C16_grabcut_slow | grabcut_rect | 0.4950 | 0.5961 | 0.5225 | 0.7936 | 0 |
| C10_grabcut_full | grabcut_rect | 0.4678 | 0.5786 | 0.4755 | 0.8928 | 0 |
| C01_hsv_pale | hsv | 0.2572 | 0.3389 | 0.2658 | 0.6170 | 0 |
| C15_lab_single | lab_center | 0.2302 | 0.2967 | 0.2817 | 0.4366 | 0 |
| C05_lab_tight | lab_center | 0.2221 | 0.3089 | 0.2943 | 0.4416 | 0 |
| C12_hybrid_lab | hybrid_lab | 0.2079 | 0.2659 | 0.2248 | 0.4258 | 0 |
| C06_lab_medium | lab_center | 0.2035 | 0.2710 | 0.2294 | 0.4461 | 0 |
| C14_hsv_single | hsv | 0.1603 | 0.1902 | 0.1634 | 0.2895 | 0 |
| C07_lab_broad | lab_center | 0.1382 | 0.1865 | 0.1463 | 0.3337 | 0 |
| C11_hybrid_hsv | hybrid_hsv | 0.1354 | 0.1723 | 0.1362 | 0.3040 | 0 |
| C02_hsv_medium | hsv | 0.1148 | 0.1550 | 0.1161 | 0.2895 | 0 |
| C13_hybrid_warm | hybrid_hsv | 0.0802 | 0.1041 | 0.0803 | 0.2040 | 0 |
| C04_hsv_warm | hsv | 0.0582 | 0.0824 | 0.0583 | 0.1803 | 0 |
| C03_hsv_broad | hsv | 0.0526 | 0.0729 | 0.0527 | 0.1585 | 0 |


## 5. Results

| Split / aggregation | IoU | Dice | Precision | Recall |
|---|---|---|---|---|
| DEV macro | 0.5107 | 0.5975 | 0.6367 | 0.6182 |
| EVAL macro | 0.6177 | 0.7346 | 0.7487 | 0.8288 |
| EVAL micro | 0.5928 | 0.7443 | 0.7384 | 0.7503 |


Metric definitions: **IoU = TP/(TP+FP+FN)**; **Dice = 2TP/(2TP+FP+FN)**; **precision = TP/(TP+FP)**; **recall = TP/(TP+FN)**. Macro averages per-image metrics with equal image weight; micro pools pixel counts before calculating the metric. These are overlap measures, not classification accuracy.

Empty-mask metric policy: both prediction and reference empty scores 1; other zero-division cases score 0. Per-image counts, TP/FP/FN/TN and masks are retained. EVAL median IoU: **0.6955**.

### Lowest-IoU held-out cases for qualitative review

| Image | Group | IoU | Dice | Prediction area | Reference area |
|---|---|---|---|---|---|
| I019 | G08 | 0.1792 | 0.3039 | 59629 | 10739 |
| I033 | G08 | 0.2015 | 0.3354 | 62993 | 19779 |
| I017 | G07 | 0.2808 | 0.4385 | 65897 | 20237 |
| I034 | G08 | 0.3623 | 0.5319 | 76737 | 32935 |
| I038 | G07 | 0.3850 | 0.5559 | 80006 | 130366 |
| I027 | G09 | 0.4303 | 0.6017 | 46434 | 20009 |
| I030 | G08 | 0.4378 | 0.6089 | 88334 | 154961 |
| I101 | G13 | 0.5067 | 0.6726 | 54500 | 27874 |
| I104 | G13 | 0.5369 | 0.6987 | 79614 | 49536 |
| I016 | G06 | 0.5653 | 0.7223 | 55639 | 96959 |


Inspect the saved three-panel overlays (classical/reference/error). Report observable failure modes such as background-color confusion, occlusion, highlights, multiple-object merging or missed objects only when actually visible. Do not infer causes solely from a metric.

## 6. Full-video individual tracking

- Status: **FULL_VIDEO_TRACKED**.
- Frames processed: **1440 / 1440**.
- Tracker: **detection_gated_hsv**.
- Persistent object labels: **boiled, raw**.
- Candidate segments: **2**; not-detected object-frames: **205**; out-of-scene object-frames: **870**.
- Detection-gated tracking uses HSV/contour candidates, shape/size gates, one-to-one temporal association and green-card suppression. not_detected and out_of_scene rows have blank coordinates; boxes are hidden rather than drifted or interpolated. OpenCV output has no audio.
- Selected annotated file: **video_tracked_detection_v2.mp4**; verified container frames: **1440**; FPS: **24.0**; resolution: **1920×1080**.
- Annotated-file SHA-256: `2716d27c26ac92168cbf7f4b10b1ccef307a04d0234e705eca73832c673c3689`. The MP4 stays on Drive, not in GitHub; this hash identifies the file that must be linked.
- Cut detection and tracking can fail under occlusion, abrupt motion or ambiguous identity; inspect the complete rendered video and disclose observed issues. Execution counts are not proof of visual correctness or tracking accuracy.

**Public annotated-video link:** `https://drive.google.com/file/d/1dbA3VdnHSSiAlocSz3NPuhAUG_yzXlcC/view?usp=sharing`

## 7. Reproducibility

Use a Colab GPU runtime for SAM. Mount Drive, place scripts in `scripts/`, and run in order with the project root `/content/drive/MyDrive/Egg_CV_Task` (the delivered scripts default to that root):

```python
from google.colab import drive
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
%run scripts/10_build_submission.py --final --video-link 'https://drive.google.com/file/d/1dbA3VdnHSSiAlocSz3NPuhAUG_yzXlcC/view?usp=sharing' --github-link 'https://github.com/lazysloth622-lab/egg-cv-segmentation-tracking'
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

- Public GitHub repository: `https://github.com/lazysloth622-lab/egg-cv-segmentation-tracking`
- Public Drive annotated video: `https://drive.google.com/file/d/1dbA3VdnHSSiAlocSz3NPuhAUG_yzXlcC/view?usp=sharing`
- Original assignment: CV Intern Task Wellfound.pdf

**Local package status:** COMPLETE_LOCAL_RESULTS. Public accessibility and final submission remain manual checks.
