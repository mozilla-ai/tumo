"""
config.py
=========

A single, central place for the settings that the other scripts share.

Keeping constants here (instead of copy-pasting them into every script) means:
  * there is ONE obvious place to change a value, and
  * every script automatically stays consistent with the others.

Nothing in this file does any real work; it only *describes* things.
Read this file first when you want to understand the project.
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# 1. The classes our model knows about
# ---------------------------------------------------------------------------
# IMPORTANT: the ORDER matters. YOLO labels each object with a number (0, 1, 2)
# and the dataset's `data.yaml` defines which number means which class:
#
#       0 -> Paper
#       1 -> Rock
#       2 -> Scissors
#
# We copy that exact order here so that `CLASS_NAMES[class_id]` always gives the
# correct human-readable name. Do NOT alphabetise this list.
CLASS_NAMES: list[str] = ["Paper", "Rock", "Scissors"]


# ---------------------------------------------------------------------------
# 2. Where things live on disk
# ---------------------------------------------------------------------------
# `Path(__file__).resolve().parent` is the folder that contains THIS file,
# i.e. the project root. Building paths relative to it means the scripts work
# no matter what directory you run them from.
PROJECT_ROOT: Path = Path(__file__).resolve().parent

# The dataset zip we were given (downloaded from Roboflow).
DATASET_ZIP: Path = Path.home() / "Downloads" / "rock-paper-scissors.v14i.yolov11.zip"

# Where `prepare_dataset.py` will unpack the images and labels.
DATASET_DIR: Path = PROJECT_ROOT / "dataset"

# The dataset description file that YOLO reads to find the train/val/test images.
DATA_YAML: Path = DATASET_DIR / "data.yaml"

# Ultralytics writes training runs (weights, plots, metrics) under here by default.
RUNS_DIR: Path = PROJECT_ROOT / "runs"


# ---------------------------------------------------------------------------
# 3. Sensible defaults for training / inference
# ---------------------------------------------------------------------------
# These are starting points tuned for teaching, not competition. Every script
# also exposes them as command-line flags so students can experiment.

# The pretrained model we start from. "yolo11n" is the *nano* size: the smallest
# and fastest YOLO11, which is exactly what we want for real-time detection on a
# Raspberry Pi 5. The first time you train, Ultralytics downloads this file.
BASE_MODEL: str = "yolo11n.pt"

# All images in this dataset are 640x640, so we train and infer at that size.
IMG_SIZE: int = 640

# How many times the model sees the whole training set. More epochs -> more
# learning (up to a point), but slower. 60 is a reasonable teaching default.
EPOCHS: int = 60

# How many images the GPU processes at once. -1 lets Ultralytics pick a value
# that fits in memory automatically — handy on the Mac's shared memory.
BATCH: int = -1

# During the game, a detection must be at least this confident (0-1) to count.
# Anything below this is treated as "no hand detected" -> the game shows "error".
CONF_THRESHOLD: float = 0.50


# ---------------------------------------------------------------------------
# 4. Picking the hardware to run on
# ---------------------------------------------------------------------------
def get_device() -> str:
    """Return the best available compute device for PyTorch as a short string.

    Ultralytics understands these strings directly:
        "mps"  -> Apple Silicon GPU (Metal) — what we use on this Mac
        "cuda" -> NVIDIA GPU (e.g. a cloud machine or Colab)
        "cpu"  -> no GPU; works everywhere but is slowest

    We import torch *inside* the function so that simply importing `config`
    (for example from the pure-Python game logic) does not require torch to be
    installed.
    """
    try:
        import torch
    except ImportError:
        # torch isn't installed (e.g. on a minimal setup) — fall back to CPU.
        return "cpu"

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# 5. Finding the weights produced by training
# ---------------------------------------------------------------------------
# Training writes its best model to runs/detect/rps_train/weights/best.pt
# (or rps_train2, rps_train3, ... if you train more than once). This helper
# returns the newest one so evaluate.py / export.py / play.py can default to it.
def default_weights() -> Path:
    """Return the path to the most recently trained ``best.pt`` (newest run).

    Falls back to the canonical first-run location if nothing is found yet, so
    error messages point somewhere sensible.
    """
    detect_dir = RUNS_DIR / "detect"
    candidates = sorted(detect_dir.glob("rps_train*/weights/best.pt"),
                        key=lambda p: p.stat().st_mtime)
    if candidates:
        return candidates[-1]  # the last one is the most recently modified
    return detect_dir / "rps_train" / "weights" / "best.pt"
