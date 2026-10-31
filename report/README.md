# Egg CV — Segmentation and Tracking

Local result status: **COMPLETE_LOCAL_RESULTS**.

Pretrained-SAM reference annotations, DEV-only classical experiments, a frozen held-out evaluation and full-video individual tracking. No SAM training is claimed. References are human-reviewed model-derived silver masks, not independent gold.

## Reviewer quick path — no execution needed

1. Open **`report/REPORT.pdf`** for methods, metrics, limitations and public links.
2. Open the **public annotated Drive video**: https://drive.google.com/file/d/1dbA3VdnHSSiAlocSz3NPuhAUG_yzXlcC/view?usp=sharing
3. Inspect `results/`, `experiments/`, `references/` and the saved evaluation overlays.
4. Public repository: https://github.com/lazysloth622-lab/egg-cv-segmentation-tracking

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
git clone https://github.com/lazysloth622-lab/egg-cv-segmentation-tracking
cd egg-cv-segmentation-tracking
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
