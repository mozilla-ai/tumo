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
  prepare_dataset.py -> train.py -> evaluate.py -> export.py -> play.py
   (unpack & fix)       (learn)      (measure)    (make fast)   (play!)
```

| Script               | What it does                                                            |
|----------------------|-------------------------------------------------------------------------|
| `config.py`          | Shared settings: class names, paths, defaults, device selection.        |
| `game_logic.py`      | The pure rules of the game (no ML, no camera). Run it to self-test.     |
| `prepare_dataset.py` | Unzips the dataset and writes a clean `data.yaml`.                      |
| `train.py`           | Trains `yolo11n` on the images (uses the Mac's GPU via MPS).            |
| `evaluate.py`        | Measures accuracy (mAP, per-class) and saves example predictions.       |
| `export.py`          | Converts the trained model to **NCNN** for fast inference on the Pi.    |
| `play.py`            | Captures a burst of frames, votes on your hand, plays a random move, declares a winner. |
| `detect_live.py`     | Live web viewer of the detections, for debugging (Mac webcam or Pi camera). |

---

## Where each step runs

This project spans **two machines**:

* **On the Mac** — prepare the data, train, evaluate and export the model
  (steps 1–5). You can also test the game with the laptop webcam.
* **On the Raspberry Pi 5** — run the game (`play.py`) and the live debugger
  (`detect_live.py`) with the real camera (steps 6–7).

Crucially, the two have **different environment setups**: the Mac uses a
uv-managed Python, while the Pi reuses its *system* Python so the venv can see the
pre-installed camera libraries. Follow **§1 for the Mac** and **§6 (step 2) for
the Pi** — don't run the Mac setup on the Pi or the camera won't import.

---

## 1. Setup — on the Mac (with `uv`)

> **This section is for the Mac.** The **Pi is set up differently** (it needs the
> system Python so picamera2/libcamera work) — see **§6, step 2**.

This project uses [`uv`](https://docs.astral.sh/uv/) to manage **both** the
Python version and the libraries. The system Python (3.14) is too new for
PyTorch/Ultralytics, so on the Mac `uv` installs a managed **Python 3.13** (see
`.python-version`). The Pi uses its own *system* Python instead (3.13 on current
Raspberry Pi OS, 3.11 on older Bookworm); the project supports **3.11–3.13** and every
dependency ships wheels for all of them, so the code behaves the same everywhere.

```bash
cd ~/workspace/tumo/rps
uv sync          # creates the virtual env and installs everything
```

`uv sync` reads `pyproject.toml`, downloads Python 3.13 if needed, and installs
Ultralytics (which brings PyTorch, OpenCV, NumPy, …) plus the NCNN export tools.

You never need to "activate" the environment — just prefix commands with
`uv run`, e.g. `uv run train.py`.

> **Workshop notebook:** `workshop.ipynb` walks through steps 2–4 interactively
> (browse the dataset, run a short training, compare models, play a round
> against a photo). Launch it with
> `uv run --with jupyter jupyter lab workshop.ipynb`.

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
uv run evaluate.py --split val
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

1. Copy the NCNN folder to the Pi, into the same `runs/…` location it has on the
   Mac so `play.py` finds it automatically (no `--weights` needed later):
   ```bash
   ssh pi@raspberrypi.local 'mkdir -p ~/workspace/tumo/rps/runs/detect/rps_train/weights'
   scp -r runs/detect/rps_train/weights/best_ncnn_model \
       pi@raspberrypi.local:~/workspace/tumo/rps/runs/detect/rps_train/weights/
   ```
2. On the Pi, build the environment. picamera2 + libcamera ship pre-installed on
   Raspberry Pi OS for the **system Python** (3.13 on current releases / Trixie, 3.11 on
   older Bookworm), so we base the venv on that interpreter and let it see the system
   packages (no compiling libcamera):
   ```bash
   sudo apt update
   sudo apt install -y python3-picamera2          # usually already present
   cd ~/workspace/tumo/rps
   export UV_PYTHON_PREFERENCE=only-system         # use the system Python, don't download one
   uv venv --python /usr/bin/python3 --system-site-packages   # /usr/bin/python3 = whatever the Pi ships
   uv sync                                         # CPU torch, ultralytics, simplejpeg, …
   ```
   `--system-site-packages` lets the venv reuse the apt picamera2/libcamera, while
   `uv sync` adds the CPU-only PyTorch and a NumPy-2.x `simplejpeg` that shadows the
   older apt one (so the camera and ML stacks agree on NumPy 2.x — otherwise picamera2
   fails to import next to torch).

   > **Don't drop `--python /usr/bin/python3`.** It must be the *system*
   > interpreter — that is the only one that can see the apt camera packages.
   > `--system-site-packages` only exposes the *base* interpreter's packages, so if
   > you let `uv` download its own Python the camera libs stay invisible and
   > `import picamera2` fails (verified on this Pi). `UV_PYTHON_PREFERENCE=only-system`
   > guards against that by forbidding a downloaded interpreter.
   >
   > **And if you ever delete `.venv`, re-run this `uv venv …` line before
   > `uv sync`.** A bare `uv sync` on its own recreates the venv *without*
   > `--system-site-packages`, so the camera silently stops importing.
3. Run the game. The NCNN export and the Pi camera are now the defaults, so no
   flags are needed:
   ```bash
   uv run play.py
   ```

**How a round works:** press *Enter*, a `3 · 2 · 1 · shoot!` countdown gives you
time to make a shape, then the camera grabs a short **burst of frames** (default
5, set with `--frames`) and the model *votes* — the class seen in the most frames
wins (ties broken by confidence). The computer then plays a random move. If no
class is seen confidently (above `--conf`, default `0.50`) in at least half the
frames, it shows **error** and lets you try again. Type `q` then *Enter* to quit.
Voting across several frames is much steadier than trusting one, possibly blurry,
shot.

> **Why does `play.py` default to `picamera2`?** Because that is what runs on the
> Pi, the real target. The `--camera opencv` option exists purely so students can
> develop and test on a laptop.

---

## 7. Debug live (see what the model sees)

When the game keeps saying "error" and you cannot tell why, `detect_live.py`
streams the camera with the detections drawn on top, viewable in any web browser
— ideal for the headless Pi (open it from your Mac).

```bash
# On the Pi (uses the NCNN model + Pi camera by default):
uv run detect_live.py
#   then browse to   http://raspberrypi.local:8000

# On the Mac, using the built-in webcam:
uv run detect_live.py --camera opencv
#   then browse to   http://localhost:8000
```

Each detection box is coloured by whether it would count in the real game:

* **green** — confidence ≥ the game threshold (`--play-conf`, default `0.50`): this *would* be played.
* **orange** — detected but below that threshold: the game ignores it and shows "error".

Because it also draws boxes well below the threshold (`--conf`, default `0.25`),
you can see *why* a hand fails — e.g. "Scissors 0.34" in orange means the model
sees it but is not confident enough, usually a lighting, distance or angle issue.
The header shows live FPS and the current verdict. Press Ctrl-C to stop.

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
tumo/rps/
├── README.md            ← you are here
├── pyproject.toml       ← uv project: Python 3.11–3.13 + dependencies
├── .python-version      ← pins Python 3.13
├── config.py            ← shared settings
├── game_logic.py        ← pure game rules (self-testing)
├── prepare_dataset.py   ← unpack + fix the dataset
├── train.py             ← train the model
├── evaluate.py          ← measure accuracy
├── export.py            ← convert to NCNN for the Pi
├── play.py              ← the game loop
├── detect_live.py       ← live detection viewer (web UI) for debugging
├── workshop.ipynb       ← interactive workshop tour (dataset → train → eval → inference)
├── dataset/             ← created by prepare_dataset.py
└── runs/                ← created by train/evaluate/export (weights, plots)
```
