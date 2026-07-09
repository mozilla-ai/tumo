"""
export.py
=========

Step 4 of the workflow: convert the trained model into a format that runs fast
on the Raspberry Pi 5.
Աշխատանքի 4-րդ քայլը՝ փոխակերպել մարզված մոդելը այնպիսի ձևաչափի, որ արագ
աշխատի Raspberry Pi 5-ի վրա:

Why convert at all? / Ինչո՞ւ ընդհանրապես փոխակերպել:
-------------------
Training produces a PyTorch model (``best.pt``). PyTorch is great on a laptop
with a GPU, but on a small ARM board like the Pi it is relatively slow. NCNN is
a lightweight neural-network engine built for exactly these devices, and
Ultralytics recommends it as the fastest option on a Raspberry Pi 5 — typically
a couple of times faster than plain PyTorch.

Մարզումն արտադրում է PyTorch մոդel (``best.pt``): PyTorch-ը հիանալի է GPU-ով
նոութբուքի վրա, բայց Pi-ի նման փոքր ARM տախտակի վրա այն համեմատաբար դանդաղ է:
NCNN-ը թեթև նեյրոնային ցանցի շարժիչ է՝ ստեղծված հատկապես այս սարքերի համար, և
Ultralytics-ը խորհուրդ է տալիս այն որպես ամենաարագ տարբերակը Raspberry Pi 5-ի
վրա՝ սովորաբար մի քանի անգամ ավելի արագ, քան սովորական PyTorch-ը:

The conversion produces a *folder* (``best_ncnn_model/``) containing the model
in NCNN's format. You copy that whole folder to the Pi and point ``play.py`` at it.

Փոխակերպումն արտադրում է *պանակ* (``best_ncnn_model/``), որը պարունակում է
մոդելը NCNN-ի ձևաչափով: Դուք ամբողջ պանակը պատճենում եք Pi-ի վրա և ``play.py``-ն
ուղղորդում դեպի այն:

Run it with / Աշխատեցրեք այսպես.
    uv run export.py                       # converts the newest best.pt
                                           # փոխակերպում է նորագույն best.pt-ն
    uv run export.py --weights path/to.pt  # convert a specific model
                                           # փոխակերպել կոնկրետ մոդel
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO

import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained YOLO11 model to NCNN for the Raspberry Pi.")
    parser.add_argument("--weights", default=None,
                        help="Model to export (default: newest trained best.pt).")
    parser.add_argument("--imgsz", type=int, default=config.IMG_SIZE,
                        help="Image size the exported model expects (default: %(default)s).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    weights = args.weights or config.default_weights()
    if not config.Path(weights).exists():
        raise SystemExit(
            f"Model weights not found at:\n  {weights}\n"
            "Train a model first with `uv run train.py`, or pass --weights."
        )

    print(f"Exporting {weights} to NCNN format...")

    model = YOLO(str(weights))

    # `.export()` does the conversion. format="ncnn" is the Pi-friendly engine.
    # It returns the path to the generated NCNN model folder.
    exported_path = model.export(format="ncnn", imgsz=args.imgsz)

    print("\nExport finished! 🎉")
    print(f"NCNN model created at:\n  {exported_path}")
    print(
        "\nNext: copy that whole folder to the Raspberry Pi, for example:\n"
        f"  scp -r '{exported_path}' pi@raspberrypi.local:~/yolo/\n"
        "then on the Pi run:\n"
        "  python play.py --weights ~/yolo/best_ncnn_model --camera picamera2"
    )


if __name__ == "__main__":
    main()
