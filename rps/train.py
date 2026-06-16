"""
train.py
========

Step 2 of the workflow: teach a YOLO11 model to recognise rock, paper and
scissors hands.

How training works in one paragraph
-----------------------------------
We start from ``yolo11n.pt``, a small model that already knows a lot about
everyday images (this is called *transfer learning*). We then show it our
labelled rock/paper/scissors pictures over and over (each full pass is an
"epoch"). After each pass the model adjusts its internal numbers so its guesses
get closer to the labels. When training finishes, Ultralytics saves the version
that scored best on the validation set as ``best.pt``.

Run it with:
    uv run train.py                       # uses the defaults from config.py
    uv run train.py --epochs 100          # train longer
    uv run train.py --epochs 1            # quick smoke test that it all works

The trained weights are written to:
    runs/detect/train*/weights/best.pt
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO

import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO11 on the rock-paper-scissors dataset.")
    # Every default comes from config.py, so the help text stays truthful even
    # if you change the values there.
    parser.add_argument("--model", default=config.BASE_MODEL,
                        help="Pretrained model to start from (default: %(default)s).")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS,
                        help="Number of passes over the training set (default: %(default)s).")
    parser.add_argument("--batch", type=int, default=config.BATCH,
                        help="Images per step; -1 lets Ultralytics auto-pick (default: %(default)s).")
    parser.add_argument("--imgsz", type=int, default=config.IMG_SIZE,
                        help="Image size in pixels (default: %(default)s).")
    parser.add_argument("--device", default=config.get_device(),
                        help="Compute device: mps / cuda / cpu (default: auto-detected).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume the most recent unfinished training run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Make sure the dataset has been prepared before we try to train on it.
    if not config.DATA_YAML.exists():
        raise SystemExit(
            f"Dataset description not found at {config.DATA_YAML}.\n"
            "Run `uv run prepare_dataset.py` first."
        )

    print(f"Starting training on device: {args.device}")
    print(f"  model={args.model}  epochs={args.epochs}  imgsz={args.imgsz}  batch={args.batch}\n")

    # Load the starting model. If `args.model` is a name like "yolo11n.pt" that
    # isn't on disk yet, Ultralytics downloads it automatically the first time.
    model = YOLO(args.model)

    # The actual training call. Each argument:
    #   data    -> our dataset description (tells YOLO where the images are)
    #   epochs  -> how many full passes over the training images
    #   imgsz   -> the size images are resized to before going into the model
    #   batch   -> how many images are processed together each step
    #   device  -> which hardware to use (mps on this Mac)
    #   patience-> stop early if validation hasn't improved for this many epochs
    #   project/name -> where results are saved: runs/detect/rps_train/
    #   resume  -> continue a previous run instead of starting fresh
    model.train(
        data=str(config.DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=20,
        project=str(config.RUNS_DIR / "detect"),
        name="rps_train",
        resume=args.resume,
    )

    # After training, point the student at the file they will actually use next.
    best_weights = config.RUNS_DIR / "detect" / "rps_train" / "weights" / "best.pt"
    print("\nTraining finished! 🎉")
    print(f"Best weights saved to:\n  {best_weights}")
    print("\nNext steps:")
    print("  uv run evaluate.py     # measure how good the model is")
    print("  uv run export.py       # convert it to NCNN for the Raspberry Pi")


if __name__ == "__main__":
    main()
