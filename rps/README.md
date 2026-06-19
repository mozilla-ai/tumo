# Rock · Paper · Scissors with YOLO11 🪨 📄 ✂️

A small, teaching-friendly project that trains a **YOLO11** object-detection model
to recognise a hand showing *rock*, *paper* or *scissors*, then plays the game
against you on a **Raspberry Pi 5** with the camera module v1.3.

The code is split into short, single-purpose scripts with lots of comments so it
can be read top-to-bottom and used to teach students each stage of a real
computer-vision project.

---

## The big picture

```
   dataset (Roboflow)                          Raspberry Pi 5 + camera
        │                                              ▲
        ▼                                              │ copy NCNN model
  prepare_dataset.py ─▶ train.py ─▶ evaluate.py ─▶ export.py ─▶ play.py
   (unpack & fix)      (learn)     (measure)      (make fast)   (play!)
```

| Script               | What it does                                                            |
|----------------------|-------------------------------------------------------------------------|
| `config.py`          | Shared settings: class names, paths, defaults, device selection.        |
| `game_logic.py`      | The pure rules of the game (no ML, no camera). Run it to self-test.      |
| `prepare_dataset.py` | Unzips the dataset and writes a clean `data.yaml`.                       |
| `train.py`           | Trains `yolo11n` on the images (uses the Mac's GPU via MPS).             |
| `evaluate.py`        | Measures accuracy (mAP, per-class) and saves example predictions.       |
| `export.py`          | Converts the trained model to **NCNN** for fast inference on the Pi.    |
| `play.py`            | Captures a frame, detects your hand, plays a random move, declares a winner. |

---

## 1. Setup (on the Mac, with `uv`)

This project uses [`uv`](https://docs.astral.sh/uv/) to manage **both** the
Python version and the libraries. The system Python (3.14) is too new for
PyTorch/Ultralytics. We pin **Python 3.11** (see `.python-version`) — the SAME
interpreter the Pi uses, so both machines resolve the same wheels and the code
behaves identically. (On the Pi, 3.11 is also what the system picamera2/libcamera
packages are built for; see step 6.)

```bash
cd /Users/mala/workspace/yolo
uv sync          # creates the virtual env and installs everything
```

`uv sync` reads `pyproject.toml`, downloads Python 3.11 if needed, and installs
Ultralytics (which brings PyTorch, OpenCV, NumPy, …) plus the NCNN export tools.

You never need to "activate" the environment — just prefix commands with
`uv run`, e.g. `uv run train.py`.

---

## 2. Prepare the dataset

```bash
uv run prepare_dataset.py
```

This unzips `~/Downloads/rock-paper-scissors.v14i.yolov11.zip` into `dataset/`
and rewrites `dataset/data.yaml` with an absolute path so YOLO can find the
images. You should see:

```
  train :  6455 images
  valid :   576 images
  test  :   304 images
```

> The three classes, in the exact order YOLO numbers them, are
> **`0 Paper`, `1 Rock`, `2 Scissors`** — taken straight from the dataset.

---

## 3. Train the model

```bash
uv run train.py                 # full training (60 epochs by default)
uv run train.py --epochs 1      # quick smoke test that everything runs
```

Training automatically uses the Apple Silicon GPU (`mps`). The best model is
saved to:

```
runs/detect/rps_train/weights/best.pt
```

Useful flags: `--epochs`, `--batch`, `--imgsz`, `--device`, `--resume`
(see `uv run train.py --help`).

---

## 4. Evaluate the model

```bash
uv run evaluate.py              # scores the test split (unseen images)
uv run evaluate.py --split valid
```

Prints the headline accuracy (mAP50, mAP50-95, precision, recall), a per-class
breakdown, and saves a confusion matrix, PR curves and a few annotated example
images under `runs/detect/`. Looking at those images is the best way for
students to *see* what the model learned.

---

## 5. Export for the Raspberry Pi (NCNN)

```bash
uv run export.py
```

Creates a folder `runs/detect/rps_train/weights/best_ncnn_model/`. NCNN is the
inference engine Ultralytics recommends on the Pi 5 — it is noticeably faster
than running the PyTorch model directly.

---

## 6. Play the game

**Test on the Mac first** (uses the built-in webcam, no Pi required):

```bash
uv run play.py --camera opencv
```

**On the Raspberry Pi 5:**

1. Copy the NCNN folder to the Pi:
   ```bash
   scp -r runs/detect/rps_train/weights/best_ncnn_model pi@raspberrypi.local:~/workspace/tumo/rps/
   ```
2. On the Pi, build the environment. picamera2 + libcamera ship pre-installed on
   Raspberry Pi OS for the **system Python 3.11**, so we base the venv on that
   interpreter and let it see the system packages (no compiling libcamera):
   ```bash
   sudo apt update
   sudo apt install -y python3-picamera2          # usually already present
   cd ~/workspace/tumo/rps
   export UV_PYTHON_PREFERENCE=only-system         # use system 3.11, do not download one
   uv venv --python /usr/bin/python3.11 --system-site-packages
   uv sync                                         # CPU torch, ultralytics, simplejpeg, …
   ```
   `--system-site-packages` lets the venv reuse the apt picamera2/libcamera, while
   `uv sync` adds the CPU-only PyTorch and a NumPy-2.x `simplejpeg` that shadows the
   older apt one (so the camera and ML stacks agree on NumPy 2.x — otherwise picamera2
   fails to import next to torch).
3. Run the game:
   ```bash
   uv run play.py --weights ~/workspace/tumo/rps/best_ncnn_model --camera picamera2
   ```

**How a round works:** press *Enter*, a `3 · 2 · 1 · shoot!` countdown gives you
time to make a shape, the camera grabs one frame, the model picks the most
confident detection, and the computer plays a random move. If no hand is seen
clearly (confidence below `--conf`, default `0.50`), it shows **error** and lets
you try again. Type `q` then *Enter* to quit.

> **Why does `play.py` default to `picamera2`?** Because that is what runs on the
> Pi, the real target. The `--camera opencv` option exists purely so students can
> develop and test on a laptop.

---

## Teaching notes / things to try

- **Change the confidence threshold** (`CONF_THRESHOLD` in `config.py`, or
  `play.py --conf`) and watch how often the model says "error".
- **Train for more or fewer epochs** and re-run `evaluate.py` to see the accuracy
  change — a great illustration of *learning*.
- **Read `game_logic.py`** first with students: it has no dependencies and
  `uv run game_logic.py` runs a built-in correctness check.
- **Per-class scores** in `evaluate.py` show that a model can be good at one hand
  shape and weak at another — a nice prompt to discuss collecting more data.

---

## Project layout

```
yolo/
├── README.md            ← you are here
├── pyproject.toml       ← uv project: Python 3.11 + dependencies
├── .python-version      ← pins Python 3.11
├── config.py            ← shared settings
├── game_logic.py        ← pure game rules (self-testing)
├── prepare_dataset.py   ← unpack + fix the dataset
├── train.py             ← train the model
├── evaluate.py          ← measure accuracy
├── export.py            ← convert to NCNN for the Pi
├── play.py              ← the game loop
├── dataset/             ← created by prepare_dataset.py
└── runs/                ← created by train/evaluate/export (weights, plots)
```
