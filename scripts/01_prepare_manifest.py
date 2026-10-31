"""Prepare the egg dataset for the CV internship assignment.

- Copies the selected dataset folder into Google Drive (without overwriting changes).
- Creates one inventory row per image, with IDs, dimensions, and SHA-256 hashes.
- Leaves scene_group_id blank for later manual review; does NOT split the dataset.

Designed for Google Colab. Developed with Arena.ai assistance; review it and
disclose AI assistance in the final assignment submission as required.
"""

from pathlib import Path
import hashlib
import re
import shutil
import pandas as pd
from PIL import Image

# Paste the dataset folder path when prompted.
source = Path(input("Paste dataset folder path: ").strip().strip('"').strip("'"))
project = Path("/content/drive/MyDrive/Egg_CV_Task")
raw = project / "data" / "raw"
manifest = project / "metadata" / "manifest.csv"
splits = ("train", "valid", "test")

if not Path("/content/drive/MyDrive").is_dir():
    raise RuntimeError("Google Drive is not mounted. Run your Drive mount cell first.")
if not all((source / s).is_dir() for s in splits):
    raise ValueError("Select the folder containing train, valid and test.")

def image_files(folder):
    return [p for split in splits for p in (folder / split).rglob("*")
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

source_images = image_files(source)
if len(source_images) != 110:
    raise ValueError(f"Expected 110 images, found {len(source_images)}. Share this output.")
if source.resolve() != raw.resolve() and raw.is_relative_to(source):
    raise ValueError("Select the dataset folder itself, not its parent project folder.")

if not raw.exists():
    raw.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, raw)
    print("Dataset copied safely to Drive.")
else:
    raw_images = image_files(raw)
    source_rel = {p.relative_to(source).as_posix() for p in source_images}
    raw_rel = {p.relative_to(raw).as_posix() for p in raw_images}
    if source_rel != raw_rel:
        raise RuntimeError("Existing raw folder differs. Nothing was overwritten; share this error.")
    if source.resolve() != raw.resolve():
        for relative in source_rel:
            if digest(source / relative) != digest(raw / relative):
                raise RuntimeError("Existing raw image differs. Nothing was overwritten; share this error.")
    print("Existing matching raw folder reused.")

def sort_key(path):
    match = re.search(r"not_damaged_(\d+)", path.name, re.I)
    return (int(match.group(1)) if match else 10**9,
            path.relative_to(raw).as_posix())

records = []
for number, path in enumerate(sorted(image_files(raw), key=sort_key), 1):
    relative = path.relative_to(raw)
    with Image.open(path) as image:
        width, height = image.size
    records.append({
        "image_id": f"I{number:03}",
        "relative_path": relative.as_posix(),
        "source_split": relative.parts[0],
        "source_class": relative.parts[1],
        "width": width,
        "height": height,
        "sha256": digest(path),
        "scene_group_id": "",
    })

df = pd.DataFrame(records)
manifest.parent.mkdir(parents=True, exist_ok=True)
if manifest.exists():
    old = pd.read_csv(manifest, keep_default_na=False)
    if (old["relative_path"].tolist() != df["relative_path"].tolist()
            or old["sha256"].tolist() != df["sha256"].tolist()):
        raise RuntimeError("Existing manifest differs. It was not overwritten.")
    df = old
    print("Existing manifest preserved.")
else:
    df.to_csv(manifest, index=False)
    print("Manifest created.")

print("Images:", len(df))
print("Exact duplicate images:", int(df["sha256"].duplicated(keep=False).sum()))
print("Manifest:", manifest)
print(df[["image_id", "source_split", "relative_path"]].head(5).to_string(index=False))
