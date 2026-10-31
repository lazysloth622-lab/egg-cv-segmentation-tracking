"""Create labeled contact sheets for manual scene-group review.

Run in Google Colab after 01_prepare_manifest.py. No extra installation needed.
This script reads the manifest and images; it does not modify either, assign
scene groups, augment images, or create a development/evaluation split.

Developed with Arena.ai assistance. Review the code and disclose AI assistance
in the final submission as required by the assignment.
"""

from pathlib import Path
import argparse
import csv
import hashlib
import re
import zipfile

from PIL import Image, ImageDraw, ImageFont


def main(project):
    project = Path(project)
    raw = project / "data" / "raw"
    manifest = project / "metadata" / "manifest.csv"
    sheet_dir = project / "metadata" / "contact_sheets"

    if not manifest.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {manifest}. Mount Drive and run Step 01 first."
        )

    # Read from disk: this script also works after a Colab runtime restart.
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"image_id", "relative_path", "source_split", "sha256"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("The manifest is missing required columns. Share this error.")
        rows = list(reader)

    if len(rows) != 110:
        raise ValueError(f"Expected 110 manifest rows, found {len(rows)}. Share this error.")
    if len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate image IDs in manifest. Share this error.")
    if len({row["relative_path"] for row in rows}) != len(rows):
        raise ValueError("Duplicate image paths in manifest. Share this error.")

    # Check all inputs before writing previews. Never silently skip an image.
    raw_root = raw.resolve()
    for row in rows:
        image_path = (raw / row["relative_path"]).resolve()
        if not image_path.is_relative_to(raw_root):
            raise ValueError(f"Image path is outside data/raw: {row['image_id']}")
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image: {image_path}")
        actual_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if actual_hash != row["sha256"]:
            raise ValueError(f"Image changed since manifest creation: {row['image_id']}")
        with Image.open(image_path) as image:
            image.verify()

    rows.sort(key=lambda row: row["image_id"])
    sheet_dir.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    sheet_paths = []
    for start in range(0, len(rows), 20):
        batch = rows[start:start + 20]
        columns, tile_width, tile_height = 4, 240, 266
        row_count = (len(batch) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * tile_width, row_count * tile_height), "white")
        draw = ImageDraw.Draw(sheet)

        for i, row in enumerate(batch):
            x = (i % columns) * tile_width
            y = (i // columns) * tile_height
            with Image.open(raw / row["relative_path"]) as image:
                image = image.convert("RGB")
                image.thumbnail((228, 228), Image.Resampling.LANCZOS)
                sheet.paste(image, (x + (tile_width - image.width) // 2, y + 4))

            match = re.search(r"not_damaged_(\d+)", Path(row["relative_path"]).name, re.I)
            source_number = match.group(1) if match else "?"
            label = f"{row['image_id']} | #{source_number} | {row['source_split']}"
            draw.text((x + 6, y + 234), label, fill="black", font=font)
            source_class = row.get("source_class", "")
            if source_class:
                draw.text((x + 6, y + 250), source_class, fill="#555555", font=font)

        output = sheet_dir / f"sheet_{start // 20 + 1:02}.jpg"
        sheet.save(output, quality=94)
        sheet_paths.append(output)

    # One convenient file to download from Drive and share for grouping review.
    zip_path = project / "metadata" / "contact_sheets.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sheet_paths:
            archive.write(path, arcname=path.name)

    assigned = sum(bool(row.get("scene_group_id", "").strip()) for row in rows)
    print("Images verified:", len(rows))
    print("Contact sheets created:", len(sheet_paths))
    print("Contact-sheet folder:", sheet_dir)
    print("ZIP to download:", zip_path)
    print(f"Scene groups assigned in manifest: {assigned}/{len(rows)}")
    print("Grouping still needs manual review. No split has been created.")

    try:
        from IPython.display import display
        print("Preview of sheet 01:")
        with Image.open(sheet_paths[0]) as preview:
            display(preview.copy())
    except ImportError:
        print("Open sheet_01.jpg to preview the first sheet.")

    return sheet_paths, zip_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", default="/content/drive/MyDrive/Egg_CV_Task",
        help="Project folder containing data/raw and metadata/manifest.csv.",
    )
    args = parser.parse_args()
    main(args.project_root)
