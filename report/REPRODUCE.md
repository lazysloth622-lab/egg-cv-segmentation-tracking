# Reproduce the experiments in Colab

## Reviewer note

You can review the PDF, public video, saved metrics, masks and prompt records without executing anything. Full manual re-annotation is optional. Do not rerun the split or final EVAL merely to obtain different numbers.

## 1. Clone and install

Open a fresh Google Colab notebook. A T4 GPU is recommended for SAM candidate generation; CPU is sufficient for classical experiments, EVAL, detection tracking and packaging.

```python
!git clone https://github.com/lazysloth622-lab/egg-cv-segmentation-tracking /content/egg_cv_repo
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
