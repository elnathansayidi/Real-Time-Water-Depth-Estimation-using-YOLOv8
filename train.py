"""
YOLOv8 Segmentation Training Script
=====================================
Requirements:
    pip install ultralytics

Usage:
    python train_yolov8_seg.py

Dataset structure expected:
    dataset/
        images/
        labels/
        data.yaml
"""

import os
import sys
import glob
from pathlib import Path
 # Removed stray line that caused an error

DATA_YAML   = "dataset/data.yaml"   # Path to your Roboflow data.yaml
MODEL       = "yolov8n-seg.pt"      # Segmentation model (auto-downloaded)
EPOCHS      = 50
IMG_SIZE    = 640
BATCH       = -1                    # -1 = auto-select safe batch size
PROJECT     = "runs/segment"        # Where results are saved
RUN_NAME    = "train"

# Number of sample images to predict on after training
NUM_SAMPLES = 5

# ── Validation helpers ────────────────────────────────────────────────────────

def check_prerequisites():
    """Verify the dataset and yaml file exist before doing anything."""
    yaml_path = Path(DATA_YAML)

    if not yaml_path.exists():
        print(f"[ERROR] data.yaml not found at: {yaml_path.resolve()}")
        print("        Make sure DATA_YAML points to your Roboflow export.")
        sys.exit(1)

    # Quick sanity-check: yaml must mention 'nc' (number of classes)
    content = yaml_path.read_text()
    if "nc:" not in content:
        print(f"[ERROR] data.yaml looks invalid – 'nc:' key not found.")
        sys.exit(1)

    dataset_dir = yaml_path.parent
    images_dir = dataset_dir / "train" / "images"
    labels_dir = dataset_dir / "train" / "labels"

    for d in (images_dir, labels_dir):
        if not d.exists():
            print(f"[ERROR] Expected directory not found: {d.resolve()}")
            sys.exit(1)

    print("[OK] Dataset structure verified.")
    print(f"     YAML  : {yaml_path.resolve()}")
    print(f"     Images: {images_dir.resolve()}")
    print(f"     Labels: {labels_dir.resolve()}")
    print()

    return yaml_path

# ── Training ──────────────────────────────────────────────────────────────────

def train(yaml_path: Path):
    from ultralytics import YOLO

    print("=" * 60)
    print("  YOLOv8 Segmentation Training")
    print("=" * 60)
    print(f"  Model   : {MODEL}")
    print(f"  Data    : {yaml_path.resolve()}")
    print(f"  Epochs  : {EPOCHS}")
    print(f"  Img size: {IMG_SIZE}")
    print(f"  Batch   : {'auto' if BATCH == -1 else BATCH}")
    print("=" * 60)
    print()

    model = YOLO(MODEL)  # Downloads yolov8n-seg.pt on first run

    results = model.train(
        data    = str(yaml_path.resolve()),
        epochs  = EPOCHS,
        imgsz   = IMG_SIZE,
        batch   = BATCH,
        project = PROJECT,
        name    = RUN_NAME,
        # Segmentation-specific – keep masks in output
        task    = "segment",
        # Useful defaults
        save    = True,       # Save best.pt + last.pt
        plots   = True,       # Save training curves
        verbose = True,
    )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    print()
    print("[DONE] Training complete.")
    print(f"       Best model saved at: {best_pt.resolve()}")
    return best_pt

# ── Validation / prediction ───────────────────────────────────────────────────

def predict(best_pt: Path, yaml_path: Path):
    from ultralytics import YOLO

    print()
    print("=" * 60)
    print("  Running predictions on sample images")
    print("=" * 60)

    # Collect sample images from the dataset
    dataset_dir = yaml_path.parent
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

    sample_images = []
    for ext in image_extensions:
        sample_images.extend(glob.glob(str(dataset_dir / "valid" / "images" / "**" / ext), recursive=True))
        if len(sample_images) >= NUM_SAMPLES:
            break

    sample_images = sample_images[:NUM_SAMPLES]

    if not sample_images:
        print("[WARN] No sample images found – skipping prediction demo.")
        return

    print(f"  Found {len(sample_images)} sample image(s).")
    print()

    model = YOLO(str(best_pt))

    pred_results = model.predict(
        source  = sample_images,
        imgsz   = IMG_SIZE,
        task    = "segment",
        save    = True,       # Saves annotated images to runs/segment/predict/
        conf    = 0.25,
        retina_masks = True,  # Higher-quality masks when possible
    )

    # ── Print per-image summary ──────────────────────────────────────────
    for i, r in enumerate(pred_results):
        img_name = Path(r.path).name
        num_dets = len(r.boxes) if r.boxes is not None else 0
        has_masks = r.masks is not None and len(r.masks) > 0

        print(f"  [{i+1}] {img_name}")
        print(f"       Detections : {num_dets}")
        print(f"       Masks ready: {'YES ✓' if has_masks else 'NO (nothing detected)'}")

        if has_masks:
            for j, mask in enumerate(r.masks.data):
                print(f"         mask[{j}] shape: {tuple(mask.shape)}")

    pred_dir = Path(pred_results[0].save_dir) if pred_results else None
    if pred_dir:
        print()
        print(f"  Annotated images saved to: {pred_dir.resolve()}")

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    yaml_path = check_prerequisites()
    best_pt   = train(yaml_path)
    predict(best_pt, yaml_path)