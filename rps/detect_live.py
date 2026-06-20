"""
detect_live.py
==============

A live debugging viewer for the Rock-Paper-Scissors model.

It opens the camera, runs the model on every frame, draws the detections, and
serves the annotated video as a tiny web page you can open in any browser — on
the Pi itself or, more usefully, from your Mac while the Pi runs headless.

Why this exists
---------------
`play.py` only announces the final move, so when the model "fails" you cannot
see why. This viewer shows EVERY detection (down to a low threshold) with its
confidence, and colours each box by whether it would actually count in the game:

    green  = confidence >= the game threshold  -> this WOULD be played
    orange = below the game threshold          -> the game ignores it ("error")

So if the game keeps saying "error", point the camera at your hand here: you will
often see something like "Scissors 0.34" in orange — proof the model *is* seeing
the hand, just not confidently enough (usually lighting, distance, or angle).

Run it
------
    # On the Pi (headless is fine — open the URL from your Mac afterwards):
    uv run detect_live.py --weights ~/workspace/tumo/rps/best_ncnn_model
    #   then browse to   http://raspberrypi.local:8000

    # On the Mac, using the built-in webcam:
    uv run detect_live.py --camera opencv
    #   then browse to   http://localhost:8000

Press Ctrl-C in the terminal to stop the server.
"""

from __future__ import annotations

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
from ultralytics import YOLO

import config
from play import open_camera  # reuse the exact camera backends play.py uses


# ===========================================================================
# Sharing the latest frame between the inference thread and the web clients
# ===========================================================================
class FrameHub:
    """A small thread-safe mailbox holding the most recent JPEG frame.

    The inference loop calls ``publish()``; each connected browser waits in
    ``next_frame()`` and is woken the moment a fresh frame is ready. A slow
    browser simply skips ahead to the latest frame — exactly what we want for a
    live preview (we never queue up stale frames).
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._seq = 0  # increments on every new frame

    def publish(self, jpeg: bytes) -> None:
        with self._cond:
            self._jpeg = jpeg
            self._seq += 1
            self._cond.notify_all()

    def next_frame(self, last_seq: int):
        """Block until a frame newer than ``last_seq`` exists, then return it."""
        with self._cond:
            while self._seq == last_seq or self._jpeg is None:
                self._cond.wait()
            return self._jpeg, self._seq


# ===========================================================================
# Drawing the detections onto a frame
# ===========================================================================
# Colours are BGR (OpenCV's order), not RGB.
GREEN = (0, 200, 0)
ORANGE = (0, 165, 255)
GREY = (60, 60, 60)
WHITE = (255, 255, 255)


def draw_overlay(frame, result, play_conf, fps, infer_ms):
    """Return a copy of the BGR ``frame`` with boxes and a status header drawn."""
    img = frame.copy()
    boxes = result.boxes

    detections = []  # collect (name, conf) for the header summary
    if boxes is not None and len(boxes) > 0:
        for i in range(len(boxes)):
            conf = float(boxes.conf[i])
            name = config.CLASS_NAMES[int(boxes.cls[i])]
            detections.append((name, conf))

            x1, y1, x2, y2 = (int(v) for v in boxes.xyxy[i].tolist())
            # Green if the game would accept it, orange if it is too unsure.
            color = GREEN if conf >= play_conf else ORANGE
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label = f"{name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)

    # ---- status header: what the *game* would decide from this exact frame ----
    playable = [d for d in detections if d[1] >= play_conf]
    if playable:
        name, conf = max(playable, key=lambda d: d[1])
        verdict, vcolor = f"WOULD PLAY: {name} ({conf:.2f})", GREEN
    elif detections:
        name, conf = max(detections, key=lambda d: d[1])
        verdict, vcolor = f"ERROR - best {name} {conf:.2f} < {play_conf:.2f}", ORANGE
    else:
        verdict, vcolor = "ERROR - nothing detected", ORANGE

    header = [
        (f"{fps:4.1f} FPS | {infer_ms:4.0f} ms/frame | play threshold {play_conf:.2f}", WHITE),
        (verdict, vcolor),
    ]
    y = 6
    for text, col in header:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (6, y), (6 + tw + 8, y + th + 10), GREY, -1)
        cv2.putText(img, text, (10, y + th + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        y += th + 14
    return img


# ===========================================================================
# The inference loop (runs in its own background thread)
# ===========================================================================
class Detector(threading.Thread):
    """Continuously capture -> detect -> annotate -> publish a JPEG."""

    def __init__(self, model, camera, hub, display_conf, play_conf, imgsz, device):
        super().__init__(daemon=True)
        self.model = model
        self.camera = camera
        self.hub = hub
        self.display_conf = display_conf
        self.play_conf = play_conf
        self.imgsz = imgsz
        self.device = device
        self.running = True

    def run(self):
        fps = 0.0
        while self.running:
            try:
                frame = self.camera.read()  # BGR numpy image
            except Exception as exc:  # camera hiccup — report once and stop
                print(f"\nCamera error: {exc}")
                self.running = False
                break

            t0 = time.time()
            results = self.model.predict(
                frame, conf=self.display_conf, imgsz=self.imgsz,
                device=self.device, verbose=False,
            )
            infer_ms = (time.time() - t0) * 1000.0

            annotated = draw_overlay(frame, results[0], self.play_conf, fps, infer_ms)
            ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                self.hub.publish(buf.tobytes())

            # Smooth the FPS read-out so it does not jump around frame to frame.
            dt = time.time() - t0
            inst = 1.0 / dt if dt > 0 else 0.0
            fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst

    def stop(self):
        self.running = False


# ===========================================================================
# The web server
# ===========================================================================
INDEX_HTML = b"""<!doctype html>
<html><head><meta charset="utf-8"><title>RPS live detection</title>
<style>
 body{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif;text-align:center}
 h1{font-size:1rem;font-weight:600;padding:.6rem;margin:0;background:#000}
 img{max-width:100vw;max-height:80vh;margin-top:.5rem;border:1px solid #333}
 .legend{margin:.7rem auto;font-size:.9rem;line-height:1.7}
 .g{color:#4caf50}.o{color:#ffa726}
</style></head>
<body>
 <h1>Rock-Paper-Scissors &mdash; live detection</h1>
 <img src="/stream.mjpg" alt="live stream">
 <div class="legend">
  <span class="g">&#9632; green</span> = confident enough to play
  &nbsp;&middot;&nbsp;
  <span class="o">&#9632; orange</span> = detected but below the game threshold (shows as &quot;error&quot;)
 </div>
</body></html>
"""


class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(INDEX_HTML)))
            self.end_headers()
            self.wfile.write(INDEX_HTML)

        elif self.path == "/stream.mjpg":
            # An MJPEG stream is just a never-ending multipart response: one JPEG
            # part per frame. Every browser understands this in a plain <img>.
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()

            hub: FrameHub = self.server.hub  # type: ignore[attr-defined]
            seq = 0
            try:
                while True:
                    jpeg, seq = hub.next_frame(seq)
                    self.wfile.write(b"--FRAME\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass  # the browser tab was closed — perfectly normal

        else:
            self.send_error(404)

    def log_message(self, *args):
        pass  # silence per-request logging; we print our own status line


# ===========================================================================
# Command-line entry point
# ===========================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live web viewer for the RPS YOLO model.")
    p.add_argument("--weights", default=None,
                   help="Model to run: a best.pt file or an NCNN folder "
                        "(default: newest NCNN export if present, else best.pt).")
    p.add_argument("--camera", default="picamera2", choices=["picamera2", "opencv"],
                   help="Camera backend (default: %(default)s; use opencv on a laptop).")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Lowest confidence to DISPLAY a box (default: %(default)s). "
                        "Kept low on purpose so you can see borderline detections.")
    p.add_argument("--play-conf", type=float, default=config.CONF_THRESHOLD,
                   help="The game's accept threshold to visualise (default: %(default)s).")
    p.add_argument("--imgsz", type=int, default=config.IMG_SIZE,
                   help="Inference image size (default: %(default)s).")
    p.add_argument("--device", default=config.get_device(),
                   help="Compute device (default: auto-detected; the Pi uses cpu).")
    p.add_argument("--host", default="0.0.0.0",
                   help="Address to bind (default: %(default)s = all interfaces).")
    p.add_argument("--port", type=int, default=8000,
                   help="Port to serve on (default: %(default)s).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    weights = args.weights or config.default_play_weights()
    if not config.Path(str(weights)).exists():
        raise SystemExit(
            f"Model not found at:\n  {weights}\n"
            "Train a model (`uv run train.py`) or pass --weights / an NCNN folder."
        )

    print(f"Loading model: {weights}")
    model = YOLO(str(weights))

    print(f"Opening camera: {args.camera}")
    camera = open_camera(args.camera)

    hub = FrameHub()
    detector = Detector(model, camera, hub, args.conf, args.play_conf,
                        args.imgsz, args.device)
    detector.start()

    server = ThreadingHTTPServer((args.host, args.port), StreamHandler)
    server.daemon_threads = True
    server.hub = hub  # type: ignore[attr-defined]

    print("\n=== Live detection viewer ===")
    print(f"Open  http://localhost:{args.port}/   (this machine)")
    print(f"  or  http://<this-host>:{args.port}/  from another computer")
    print(f"      e.g. http://raspberrypi.local:{args.port}/ when running on the Pi")
    print(f"Display >= {args.conf:.2f}, play threshold {args.play_conf:.2f}. Ctrl-C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        detector.stop()
        server.shutdown()
        camera.close()


if __name__ == "__main__":
    main()
