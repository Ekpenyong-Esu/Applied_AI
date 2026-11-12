from pathlib import Path

SEED = 42

LAB_DIR = Path(__file__).resolve().parent.parent

VECS_FILE = LAB_DIR / "vecs.npy"

# my_dataset/<category>/<photos>
MY_DATASET_DIR = LAB_DIR / "my_dataset"
FIGURES_DIR = LAB_DIR / "figures"

PHOTO_SIZE = (128, 128)
