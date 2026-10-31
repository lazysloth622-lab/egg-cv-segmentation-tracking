"""Install pinned official SAM, download ViT-B, verify GPU and load the model.

Run in a Colab GPU runtime after 04_freeze_split.py. Uses the existing Colab
PyTorch/torchvision installation; does NOT upgrade or reinstall those packages.
No model training or reference masks are produced by this setup step.

Developed with Arena.ai assistance. Run, inspect and disclose its use.
"""

from pathlib import Path
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time

SAM_COMMIT = "dca509fe793f601edb92606367a655c15ac00fdf"
SAM_PACKAGE = "git+https://github.com/facebookresearch/segment-anything.git@" + SAM_COMMIT
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
CHECKPOINT_NAME = "sam_vit_b_01ec64.pth"


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download_checkpoint(url, destination, attempts=4):
    """Resume a partial HTTPS download. Final size and SHA are recorded locally.

    The full hash is measured after download; it is not claimed to be a
    separately published official hash. Trust source: the official HTTPS URL.
    """
    import requests
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt = destination.with_suffix(destination.suffix + ".integrity.json")
    if destination.exists():
        if destination.stat().st_size < 100_000_000:
            raise RuntimeError("Existing checkpoint is unexpectedly small. Share this error; do not load it.")
        actual_hash = file_sha256(destination)
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved["sha256"] != actual_hash or saved["bytes"] != destination.stat().st_size:
                raise RuntimeError("Saved checkpoint has changed. Stop and share this error.")
            print("Existing checkpoint verified and reused.")
            return saved
        info = {"source_url": url, "bytes": destination.stat().st_size, "sha256": actual_hash,
                "hash_note": "Measured locally; no independently published full checksum was supplied."}
        receipt.write_text(json.dumps(info, indent=2) + "\n")
        print("Existing checkpoint fingerprint recorded; model loading will validate its format.")
        return info

    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(attempts):
        start = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={start}-"} if start else {}
        try:
            with requests.get(url, stream=True, headers=headers, timeout=(30, 120)) as response:
                if response.status_code == 416 and start:
                    # Do not assume a stale partial download is complete.
                    raise RuntimeError("Server rejected the partial-download offset. Share this error.")
                response.raise_for_status()
                append = bool(start and response.status_code == 206)
                if not append:
                    start = 0  # A server may ignore Range: restart, never append a full body.
                total = None
                if response.status_code == 206:
                    content_range = response.headers.get("Content-Range", "")
                    try:
                        range_start = int(content_range.split()[1].split("-")[0])
                        total = int(content_range.rsplit("/", 1)[1])
                    except (ValueError, IndexError):
                        raise RuntimeError("Invalid resume response from checkpoint server.")
                    if range_start != start:
                        raise RuntimeError("Server returned the wrong resume offset.")
                elif response.headers.get("Content-Length"):
                    total = int(response.headers["Content-Length"])
                print(f"Checkpoint download: attempt {attempt + 1}/{attempts}; starting at {start / 1e6:.1f} MB")
                last_report = start
                with partial.open("ab" if append else "wb") as handle:
                    for chunk in response.iter_content(chunk_size=2 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                            if handle.tell() - last_report >= 50 * 1024 * 1024:
                                last_report = handle.tell()
                                if total:
                                    print(f"  {last_report / 1e6:.0f}/{total / 1e6:.0f} MB")
                                else:
                                    print(f"  {last_report / 1e6:.0f} MB")
                size = partial.stat().st_size
                if total is not None and size != total:
                    raise IOError(f"Incomplete download: {size} of {total} bytes")
                if size < 100_000_000:
                    raise RuntimeError("Download is too small to be the SAM ViT-B checkpoint.")
            partial.replace(destination)
            result = {"source_url": url, "bytes": destination.stat().st_size,
                      "sha256": file_sha256(destination),
                      "hash_note": "Measured locally; no independently published full checksum was supplied."}
            receipt.write_text(json.dumps(result, indent=2) + "\n")
            print("Checkpoint download complete.")
            return result
        except (requests.RequestException, IOError) as exc:
            print(f"Download interrupted: {exc}")
            if attempt == attempts - 1:
                raise RuntimeError("Checkpoint download could not finish. The partial file is saved; share the error and retry later.") from exc
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError("Checkpoint download did not complete.")


def ensure_sam_installed():
    commit = None
    try:
        distribution = importlib.metadata.distribution("segment-anything")
        direct = distribution.read_text("direct_url.json")
        if direct:
            commit = json.loads(direct).get("vcs_info", {}).get("commit_id")
    except importlib.metadata.PackageNotFoundError:
        pass
    if commit == SAM_COMMIT:
        print("Pinned official SAM package already installed.")
        return
    if "segment_anything" in sys.modules:
        raise RuntimeError("A different SAM package is already imported. Restart the runtime, mount Drive and rerun setup.")
    print("Installing pinned official SAM package (without changing PyTorch)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
                           "--force-reinstall", SAM_PACKAGE])
    importlib.invalidate_caches()


def main(project_root):
    project = Path(project_root)
    metadata = project / "metadata"
    freeze_path = metadata / "split_freeze.json"
    if not freeze_path.is_file():
        raise RuntimeError("No frozen split found. Finish Step 04 and explicitly approve grouping first.")
    freeze = json.loads(freeze_path.read_text())
    for filename, expected in freeze["file_sha256"].items():
        path = metadata / filename
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"Frozen data file changed: {filename}. Do not resplit.")

    import torch
    print("PyTorch:", torch.__version__)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is not active. Colab: Runtime > Change runtime type > T4 GPU. Then mount Drive and rerun setup.")
    device = "cuda"
    print("GPU:", torch.cuda.get_device_name(0))
    ensure_sam_installed()
    try:
        import torchvision
        from segment_anything import SamPredictor, sam_model_registry
    except Exception as exc:
        raise RuntimeError("SAM/torchvision import failed. Share the complete error; do not randomly upgrade torch or torchvision.") from exc

    checkpoint = project / "checkpoints" / CHECKPOINT_NAME
    checkpoint_info = download_checkpoint(CHECKPOINT_URL, checkpoint)
    print("Loading SAM ViT-B onto the GPU...")
    model = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
    model.to(device=device)
    model.eval()
    predictor = SamPredictor(model)

    versions = {}
    for package in ("torch", "torchvision", "numpy", "Pillow", "opencv-python", "segment-anything"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    info = {
        "model_type": "vit_b", "sam_commit": SAM_COMMIT,
        "official_repository": "https://github.com/facebookresearch/segment-anything",
        "checkpoint": checkpoint_info,
        "checkpoint_relative_path": checkpoint.relative_to(project).as_posix(),
        "device": device, "gpu": torch.cuda.get_device_name(0),
        "python": platform.python_version(), "torch_cuda": torch.version.cuda,
        "package_versions": versions,
        "split_sha256": file_sha256(metadata / "split.csv"),
        "reference_policy": "Segment visible egg pixels only. Human boxes/points and reviewed masks; no hidden full-ellipse completion. SAM scores are not egg-identity confidence. Apply the same labeling policy to dev and eval.",
        "ai_assistance": "Setup and annotation helper developed with Arena.ai assistance. Model execution and mask review must be real and disclosed.",
    }
    environment_path = metadata / "sam_environment.json"
    if environment_path.exists():
        old = json.loads(environment_path.read_text())
        same = (old.get("sam_commit") == SAM_COMMIT
                and old.get("checkpoint", {}).get("sha256") == checkpoint_info["sha256"]
                and old.get("split_sha256") == info["split_sha256"])
        if not same:
            raise RuntimeError("An existing SAM run used different weights/code/split. Stop; do not mix reference versions.")
        environment_path.write_text(json.dumps(info, indent=2) + "\n")
    else:
        environment_path.write_text(json.dumps(info, indent=2) + "\n")
    print("\nSAM READY")
    print("Model: vit_b | training: NO | reference masks generated so far by this setup: 0")
    print("Checkpoint SHA-256:", checkpoint_info["sha256"])
    print("Environment saved:", environment_path)
    print("Keep this GPU runtime open. SAM_PREDICTOR and SAM_RUN_INFO are available for Step 06.")
    return predictor, info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default="/content/drive/MyDrive/Egg_CV_Task")
    args = parser.parse_args()
    SAM_PREDICTOR, SAM_RUN_INFO = main(args.project_root)
