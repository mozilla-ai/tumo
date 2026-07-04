"""
evaluate.py
===========

Step 3 of the workflow: measure how good the trained model actually is.

It is not enough to train a model — we need to know whether it works. This
script runs the model over a split it was scored on (the *test* set by default,
i.e. images it was never trained on) and reports standard object-detection
metrics, then saves a few annotated example images so you can *see* the results.

Key metrics, in plain language
------------------------------
  precision : of the hands the model flagged, how many were correct?
  recall    : of the hands that were really there, how many did it find?
  mAP50     : overall accuracy score (higher is better, 1.0 is perfect) using a
              lenient overlap rule. A good, easy headline number.
  mAP50-95  : a stricter average score; always lower than mAP50.

Run it with:
    uv run evaluate.py                       # uses newest best.pt, test split
    uv run evaluate.py --split val           # evaluate on the validation split
    uv run evaluate.py --weights path/to.pt  # evaluate a specific model
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO

import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO11 rock-paper-scissors model.")
    parser.add_argument("--weights", default=None,
                        help="Model to evaluate (default: newest trained best.pt).")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Which dataset split to evaluate on (default: %(default)s).")
    parser.add_argument("--device", default=config.get_device(),
                        help="Compute device: mps / cuda / cpu (default: auto-detected).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Default to the most recently trained weights if the user didn't name one.
    weights = args.weights or config.default_weights()
    if not str(weights) or not config.Path(weights).exists():
        raise SystemExit(
            f"Model weights not found at:\n  {weights}\n"
            "Train a model first with `uv run train.py`, or pass --weights."
        )

    print(f"Evaluating {weights}")
    print(f"  split={args.split}  device={args.device}\n")

    model = YOLO(str(weights))

    # `.val()` runs the model over the chosen split and computes the metrics.
    #   data  -> the dataset description (same file training used)
    #   split -> which set of images to score against
    #   plots -> also save confusion matrix + precision/recall curves as images
    # Results (numbers AND plots) are written under runs/detect/rps_eval/.
    metrics = model.val(
        data=str(config.DATA_YAML),
        split=args.split,
        device=args.device,
        plots=True,
        project=str(config.RUNS_DIR / "detect"),
        name="rps_eval",
    )

    # ----- Print the headline numbers -----
    # `metrics.box` holds the detection scores. These are averages over all classes.
    print("\nOverall results")
    print("---------------")
    print(f"  mAP50    : {metrics.box.map50:.3f}   (higher is better, max 1.0)")
    print(f"  mAP50-95 : {metrics.box.map:.3f}   (stricter average)")
    print(f"  precision: {metrics.box.mp:.3f}")
    print(f"  recall   : {metrics.box.mr:.3f}")

    # ----- Print a per-class breakdown -----
    # This shows whether the model is, say, great at "Rock" but weak at "Scissors".
    print("\nPer-class mAP50")
    print("---------------")
    # `metrics.box.maps` is one mAP50-95 value per class; `ap_class_index` tells
    # us which class each entry belongs to.
    for class_id, ap in zip(metrics.box.ap_class_index, metrics.box.maps):
        print(f"  {config.CLASS_NAMES[class_id]:<9}: {ap:.3f}")

    # ----- Save some annotated example predictions -----
    # Numbers are useful, but seeing the boxes drawn on real images is the best
    # way for students to build intuition. We predict on a handful of test images
    # and save the annotated copies under runs/detect/rps_predictions/.
    sample_dir = config.DATASET_DIR / args.split.replace("val", "valid") / "images"
    sample_images = sorted(sample_dir.glob("*.jpg"))[:8]
    if sample_images:
        print(f"\nSaving annotated predictions for {len(sample_images)} sample images...")
        model.predict(
            source=[str(p) for p in sample_images],
            device=args.device,
            conf=config.CONF_THRESHOLD,
            save=True,
            project=str(config.RUNS_DIR / "detect"),
            name="rps_predictions",
        )

    print("\nDone. Look inside runs/detect/ for the saved plots and example images.")


if __name__ == "__main__":
    main()
