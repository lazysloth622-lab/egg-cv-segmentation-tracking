# Push the generated repository to GitHub

The target public repository is: https://github.com/lazysloth622-lab/egg-cv-segmentation-tracking

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
git remote add origin https://github.com/lazysloth622-lab/egg-cv-segmentation-tracking
git push -u origin main
```

If Git asks for your identity, set your own name and GitHub email:

```bash
git config --global user.name "YOUR NAME"
git config --global user.email "YOUR GITHUB EMAIL"
```

Use the normal Git Credential Manager/browser sign-in prompt. Never paste a password or personal access token into chat, the report or the repository. Do not use force-push. These commands assume the public repository was created empty; if it already has conflicting commits, stop and resolve that safely.

## After pushing

- Open https://github.com/lazysloth622-lab/egg-cv-segmentation-tracking in an incognito/logged-out browser.
- Confirm `report/REPORT.pdf` exists and opens.
- Open the PDF's Drive-video link and confirm it plays the accepted corrected MP4 without sign-in.
- Confirm `README.md`, `report/REPRODUCE.md`, `requirements.txt`, scripts 01–10, all 16 trial logs and EVAL outputs are visible.
- Submit the public repository URL within the assignment deadline.
