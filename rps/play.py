"""
play.py
=======

Step 5 — the game itself! Capture a frame from the camera, let the model decide
whether it sees rock, paper or scissors, pick a random move for the computer,
and announce the winner.

Where it runs
-------------
  * On the Raspberry Pi 5 with the camera module v1.3 -> use --camera picamera2
    (the default). This needs the system package `python3-picamera2`.
  * On your Mac/laptop for testing -> use --camera opencv to use the built-in
    webcam. This needs no extra setup (OpenCV ships with Ultralytics).

The model can be either a PyTorch file (best.pt) or, on the Pi, the faster NCNN
folder produced by export.py — Ultralytics loads both the same way.

Run it with:
    uv run play.py --camera opencv                 # test on the Mac webcam
    python play.py --camera picamera2 \\
        --weights ~/yolo/best_ncnn_model           # on the Pi

Controls: press Enter to play a round, type 'q' then Enter to quit.
"""

from __future__ import annotations

import argparse
import time

from ultralytics import YOLO

import config
from game_logic import decide_winner, detection_to_move, random_move


# ===========================================================================
# Camera backends
# ===========================================================================
# Both backends expose the same tiny interface: `.read()` returns one frame as a
# BGR numpy image (the format OpenCV and Ultralytics expect), and `.close()`
# releases the camera. Hiding the differences here keeps the game loop simple.

class OpenCVCamera:
    """Webcam access via OpenCV — used for testing on a laptop/desktop."""

    def __init__(self, index: int = 0):
        import cv2  # imported lazily so the Pi path doesn't need it loaded first
        self._cv2 = cv2
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open webcam #{index}. Is another app using it?")

    def read(self):
        ok, frame = self._cap.read()  # frame is already BGR
        if not ok:
            raise RuntimeError("Failed to read a frame from the webcam.")
        return frame

    def close(self):
        self._cap.release()


class PiCamera:
    """Raspberry Pi camera access via Picamera2 — used on the Pi."""

    def __init__(self):
        # picamera2 is a SYSTEM package on Raspberry Pi OS, not a pip wheel, so
        # we import it lazily and give a helpful message if it is missing.
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError(
                "picamera2 is not available. On Raspberry Pi OS install it with:\n"
                "  sudo apt install -y python3-picamera2\n"
                "(Or test on a laptop with `--camera opencv`.)"
            ) from exc

        import cv2
        self._cv2 = cv2
        self._picam = Picamera2()
        # Configure a still-image capture at our model's input size.
        cfg = self._picam.create_preview_configuration(
            main={"size": (config.IMG_SIZE, config.IMG_SIZE), "format": "RGB888"}
        )
        self._picam.configure(cfg)
        self._picam.start()
        time.sleep(1.0)  # give the sensor a moment to adjust exposure

    def read(self):
        # Picamera2 hands back an RGB image; OpenCV/Ultralytics expect BGR, so
        # we flip the colour channels to keep the model's colours correct.
        rgb = self._picam.capture_array()
        return self._cv2.cvtColor(rgb, self._cv2.COLOR_RGB2BGR)

    def close(self):
        self._picam.stop()


def open_camera(kind: str):
    """Create the requested camera backend."""
    if kind == "picamera2":
        return PiCamera()
    if kind == "opencv":
        return OpenCVCamera()
    raise ValueError(f"Unknown camera type: {kind!r}")


# ===========================================================================
# Detection
# ===========================================================================
def detect_hand(model: YOLO, frame, conf_threshold: float, device: str):
    """Run the model on one frame and return the best detected class name.

    Returns ``(class_name, confidence)`` for the most confident hand found, or
    ``None`` if nothing scored above ``conf_threshold`` (our "error" case).
    """
    # `predict` returns a list (one entry per image); we passed one image.
    results = model.predict(frame, conf=conf_threshold, device=device, verbose=False)
    boxes = results[0].boxes

    # No boxes at all -> no hand detected.
    if boxes is None or len(boxes) == 0:
        return None

    # Each box has a confidence score; find the index of the most confident one.
    # `.conf` / `.cls` are tensors, so we convert single values with float()/int().
    confidences = boxes.conf.tolist()
    best_i = max(range(len(confidences)), key=lambda i: confidences[i])

    class_id = int(boxes.cls[best_i])
    confidence = float(boxes.conf[best_i])
    return config.CLASS_NAMES[class_id], confidence


# ===========================================================================
# The game loop
# ===========================================================================
def play_round(model, camera, conf_threshold, device) -> str | None:
    """Play exactly one round. Returns "win"/"lose"/"tie", or None on error."""
    # A short countdown gives the player time to form their shape.
    for n in (3, 2, 1):
        print(f"  {n}...", end=" ", flush=True)
        time.sleep(0.8)
    print("shoot!")

    frame = camera.read()
    detection = detect_hand(model, frame, conf_threshold, device)

    # No confident detection -> show the "error" outcome and let them retry.
    if detection is None:
        print("  🤖 error: I couldn't see a clear hand. Try again!")
        return None

    class_name, confidence = detection
    player_move = detection_to_move(class_name)      # e.g. "Rock" -> "rock"
    computer_move = random_move()
    outcome = decide_winner(player_move, computer_move)

    # Friendly summary of the round.
    print(f"  👋 You played : {player_move}  (confidence {confidence:.0%})")
    print(f"  🤖 I played   : {computer_move}")
    verdict = {"win": "You win! 🎉", "lose": "I win! 🤖", "tie": "It's a tie 🤝"}[outcome]
    print(f"  >> {verdict}")
    return outcome


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Rock-Paper-Scissors against the YOLO11 model.")
    parser.add_argument("--weights", default=None,
                        help="Model to play with: a best.pt file or an NCNN folder "
                             "(default: newest trained best.pt).")
    parser.add_argument("--camera", default="picamera2", choices=["picamera2", "opencv"],
                        help="Camera backend (default: %(default)s; use opencv to test on a laptop).")
    parser.add_argument("--conf", type=float, default=config.CONF_THRESHOLD,
                        help="Minimum confidence to accept a detection (default: %(default)s).")
    parser.add_argument("--rounds", type=int, default=0,
                        help="Number of rounds to play; 0 means keep playing until you quit.")
    parser.add_argument("--device", default=config.get_device(),
                        help="Compute device (default: auto-detected; the Pi uses cpu).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    weights = args.weights or config.default_weights()
    if not config.Path(str(weights)).exists():
        raise SystemExit(
            f"Model not found at:\n  {weights}\n"
            "Train a model (`uv run train.py`) or pass --weights / an NCNN folder."
        )

    print(f"Loading model: {weights}")
    model = YOLO(str(weights))

    print(f"Opening camera: {args.camera}")
    camera = open_camera(args.camera)

    # Running tally of results from the player's point of view.
    score = {"win": 0, "lose": 0, "tie": 0}

    print("\n=== Rock-Paper-Scissors vs. the computer ===")
    print("Press Enter to play a round, or type 'q' then Enter to quit.\n")

    try:
        round_number = 0
        while True:
            # Stop after a fixed number of rounds if --rounds was given.
            if args.rounds and round_number >= args.rounds:
                break

            command = input(f"Round {round_number + 1} — ready? [Enter / q] ").strip().lower()
            if command == "q":
                break

            outcome = play_round(model, camera, args.conf, args.device)
            if outcome is None:
                # Errors don't count as a round; let the player try again.
                continue

            score[outcome] += 1
            round_number += 1
            print(f"  Score so far — you {score['win']} : {score['lose']} me  "
                  f"({score['tie']} ties)\n")
    except KeyboardInterrupt:
        # Ctrl-C is a perfectly normal way to stop the game.
        print("\nInterrupted.")
    finally:
        # Always release the camera, even if something went wrong.
        camera.close()

    print("\nThanks for playing!")
    print(f"Final score — you {score['win']} : {score['lose']} me ({score['tie']} ties)")


if __name__ == "__main__":
    main()
