"""Task 1.1: load my photos (my_dataset/<category>/*.jpg) into an (n, x, y, 3) array."""

from pathlib import Path

import numpy as np
from PIL import Image

from .config import MY_DATASET_DIR, PHOTO_SIZE

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_photo(path, size=PHOTO_SIZE):
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB").resize(size), dtype=np.uint8)


def load_image_folder(root=MY_DATASET_DIR, size=PHOTO_SIZE):
    """Returns X (n, x, y, 3) uint8, y (n,) labels and the category names (index = label)."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist, put the photos in {root}/<category>/")
    names = sorted(p.name for p in root.iterdir() if p.is_dir())
    images, labels = [], []
    for label, name in enumerate(names):
        for photo in sorted((root / name).iterdir()):
            if photo.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(load_photo(photo, size))
                labels.append(label)
    return np.stack(images), np.array(labels), names


def describe_dataset(X, y, names):
    n, x, yy, c = X.shape
    lines = [f"Array shape: (n={n}, x={x}, y={yy}, channels={c}), dtype={X.dtype}"]
    for label, name in enumerate(names):
        lines.append(f"  label {label}: {name:<15} {int(np.sum(y == label))} photos")
    return "\n".join(lines)
