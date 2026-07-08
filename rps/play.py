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
folder produced by export.py — Ultralytics loads both the same way. By default we
pick the NCNN export automatically when one exists (it is much faster on the Pi).

Each round captures a short burst of frames (default 5) and takes the class seen
most often, which is much steadier than relying on a single frame.

Run it with:
    uv run play.py --camera opencv                 # test on the Mac webcam
    python play.py --camera picamera2 \\
        --weights ~/yolo/best_ncnn_model           # on the Pi

Controls: press Enter to play a round, type 'q' then Enter to quit.

═══════════════════════════════════════════════════════════════════════════
ՀԱՅԵՐԵՆ
═══════════════════════════════════════════════════════════════════════════

play.py
=======

Քայլ 5 — բուն խաղը։ Տեսախցիկից վերցնում ենք մեկ կադր, թողնում ենք մոդելին
որոշել՝ քար, թուղթ, թե մկրատ է տեսնում, համակարգչի համար ընտրում ենք պատահական
քայլ և հայտարարում ենք հաղթողին։

Որտեղ է աշխատում
---------------
  * Raspberry Pi 5-ի վրա՝ v1.3 տեսախցիկի մոդուլով -> օգտագործեք --camera picamera2
    (լռելյայն տարբերակը)։ Սա պահանջում է `python3-picamera2` համակարգային փաթեթը։
  * Ձեր Mac/նոութբուքի վրա փորձարկելու համար -> օգտագործեք --camera opencv՝
    ներկառուցված վեբ-տեսախցիկն օգտագործելու համար։ Սա հավելյալ կարգավորում
    չի պահանջում (OpenCV-ն արդեն գալիս է Ultralytics-ի հետ)։

Մոդելը կարող է լինել կա՛մ PyTorch ֆայլ (best.pt), կա՛մ Pi-ի վրա՝ ավելի արագ NCNN
թղթապանակը, որը ստեղծում է export.py-ը — Ultralytics-ը երկուսն էլ նույն ձևով է
բեռնում։ Լռելյայն մենք ինքնաշխատ ընտրում ենք NCNN տարբերակը, եթե այն առկա է
(Pi-ի վրա այն շատ ավելի արագ է)։

Ամեն ռաունդ վերցնում է մի քանի կադր անընդմեջ (լռելյայն՝ 5) և վերցնում է ամենից
հաճախ տեսնված դասը, ինչը շատ ավելի կայուն է, քան մեկ կադրի վրա հիմնվելը։

Գործարկեք հետևյալով՝
    uv run play.py --camera opencv                 # փորձարկել Mac վեբ-տեսախցիկի վրա
    python play.py --camera picamera2 \\
        --weights ~/yolo/best_ncnn_model           # Pi-ի վրա

Կառավարում՝ սեղմեք Enter՝ ռաունդ խաղալու համար, մուտքագրեք 'q' ապա Enter՝ դուրս գալու համար։
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict

from ultralytics import YOLO

import config
from game_logic import decide_winner, detection_to_move, random_move


# ===========================================================================
# Camera backends
# Տեսախցիկի հետնամասերը (backends)
# ===========================================================================
# Both backends expose the same tiny interface: `.read()` returns one frame as a
# BGR numpy image (the format OpenCV and Ultralytics expect), and `.close()`
# releases the camera. Hiding the differences here keeps the game loop simple.
#
# Երկու հետնամասերն էլ ունեն միևնույն փոքրիկ ինտերֆեյսը՝ `.read()`-ը վերադարձնում է
# մեկ կադր՝ որպես BGR numpy պատկեր (այն ձևաչափը, որ սպասում են OpenCV-ն և
# Ultralytics-ը), իսկ `.close()`-ը ազատում է տեսախցիկը։ Տարբերությունները այստեղ
# թաքցնելը պահում է խաղի ցիկլը պարզ։

class OpenCVCamera:
    """Webcam access via OpenCV — used for testing on a laptop/desktop.

    Վեբ-տեսախցիկի հասանելիությունը OpenCV-ի միջոցով — օգտագործվում է
    նոութբուքի/համակարգչի վրա փորձարկելու համար։
    """

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
    """Raspberry Pi camera access via Picamera2 — used on the Pi.

    Raspberry Pi-ի տեսախցիկի հասանելիությունը Picamera2-ի միջոցով —
    օգտագործվում է Pi-ի վրա։
    """

    def __init__(self):
        # picamera2 is a SYSTEM package on Raspberry Pi OS, not a pip wheel, so
        # we import it lazily and give a helpful message if it is missing.
        #
        # picamera2-ը Raspberry Pi OS-ի ՀԱՄԱԿԱՐԳԱՅԻՆ փաթեթ է, ոչ թե pip wheel, ուստի
        # մենք այն ներմուծում ենք ուշացումով և տալիս ենք օգտակար հաղորդագրություն,
        # եթե այն բացակայում է։
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError(
                "picamera2 is not available. On Raspberry Pi OS install it with:\n"
                "  sudo apt install -y python3-picamera2\n"
                "(Or test on a laptop with `--camera opencv`.)"
            ) from exc

        self._picam = Picamera2()
        # Configure a still-image capture at our model's input size.
        # Կարգավորում ենք ստատիկ պատկերի գրանցումը մեր մոդելի մուտքի չափով։
        cfg = self._picam.create_preview_configuration(
            main={"size": (config.IMG_SIZE, config.IMG_SIZE), "format": "RGB888"}
        )
        self._picam.configure(cfg)
        self._picam.start()
        time.sleep(1.0)  # give the sensor a moment to adjust exposure
                         # տալիս ենք սենսորին մի պահ՝ էքսպոզիցիան կարգավորելու համար

    def read(self):
        # IMPORTANT picamera2 quirk: its "RGB888" format actually returns the
        # channels in B,G,R order in memory — already exactly what OpenCV and
        # Ultralytics expect. So we return the array as-is. (Verified on hardware:
        # adding a cvtColor here SWAPS red/blue, which both tints the preview blue
        # AND feeds the model wrong colours, hurting detection accuracy.)
        #
        # ԿԱՐԵՎՈՐ picamera2 առանձնահատկություն՝ նրա "RGB888" ձևաչափն իրականում
        # վերադարձնում է ալիքները հիշողության մեջ B,G,R հերթականությամբ — հենց այն,
        # ինչ սպասում են OpenCV-ն և Ultralytics-ը։ Ուստի զանգվածը վերադարձնում ենք
        # այնպես, ինչպես կա։ (Ստուգված է սարքի վրա՝ այստեղ cvtColor ավելացնելը
        # ՏԵՂԱՓՈԽՈՒՄ Է կարմիրն ու կապույտը, ինչը և՛ կապտեցնում է նախադիտումը,
        # և՛ մոդելին սխալ գույներ է տալիս՝ վնասելով հայտնաբերման ճշգրտությանը։)
        return self._picam.capture_array()

    def close(self):
        self._picam.stop()


def open_camera(kind: str):
    """Create the requested camera backend.

    Ստեղծում է պահանջված տեսախցիկի հետնամասը (backend)։
    """
    if kind == "picamera2":
        return PiCamera()
    if kind == "opencv":
        return OpenCVCamera()
    raise ValueError(f"Unknown camera type: {kind!r}")


# ===========================================================================
# Detection
# Հայտնաբերում
# ===========================================================================
def detect_hand(model: YOLO, frame, conf_threshold: float, device: str):
    """Run the model on one frame and return the best detected class name.

    Returns ``(class_name, confidence)`` for the most confident hand found, or
    ``None`` if nothing scored above ``conf_threshold`` (our "error" case).

    ───────────────────────────────────────────────────────────────────────
    Գործարկում է մոդելը մեկ կադրի վրա և վերադարձնում լավագույն հայտնաբերված
    դասի անունը։

    Վերադարձնում է ``(class_name, confidence)`` ամենավստահ գտնված ձեռքի համար,
    կամ ``None``, եթե ոչինչ չի գերազանցել ``conf_threshold``-ը (մեր «սխալի» դեպքը)։
    """
    # `predict` returns a list (one entry per image); we passed one image.
    # `predict`-ը վերադարձնում է ցուցակ (մեկ գրառում ամեն պատկերի համար). մենք
    # փոխանցեցինք մեկ պատկեր։
    results = model.predict(frame, conf=conf_threshold, device=device, verbose=False)
    boxes = results[0].boxes

    # No boxes at all -> no hand detected.
    # Ընդհանրապես ուղղանկյուններ չկան -> ձեռք չի հայտնաբերվել։
    if boxes is None or len(boxes) == 0:
        return None

    # Each box has a confidence score; find the index of the most confident one.
    # `.conf` / `.cls` are tensors, so we convert single values with float()/int().
    #
    # Ամեն ուղղանկյուն ունի վստահության միավոր. գտնում ենք ամենավստահի ինդեքսը։
    # `.conf` / `.cls`-ը թենզորներ են, ուստի առանձին արժեքները փոխարկում ենք
    # float()/int()-ով։
    confidences = boxes.conf.tolist()
    best_i = max(range(len(confidences)), key=lambda i: confidences[i])

    class_id = int(boxes.cls[best_i])
    confidence = float(boxes.conf[best_i])
    return config.CLASS_NAMES[class_id], confidence


def detect_hand_burst(model, camera, frames, conf_threshold, device):
    """Capture several frames in quick succession and combine them by voting.

    Why not a single frame? One frame can be motion-blurred or caught mid-move,
    which makes readings noisy. Looking at a short burst and taking the class
    seen *most often* (ties broken by total confidence) is far steadier.

    Returns ``(class_name, avg_confidence, votes)`` for the winning class, where
    ``votes`` is a ``{class_name: count}`` dict, or ``None`` if no class showed up
    in at least half the frames (our "error" case — too unsure to call).

    ───────────────────────────────────────────────────────────────────────
    Վերցնում է մի քանի կադր արագ հաջորդականությամբ և միավորում դրանք
    քվեարկությամբ։

    Ինչու՞ ոչ մեկ կադր։ Մեկ կադրը կարող է շարժումից մշուշոտ լինել կամ բռնվել
    շարժման կեսին, ինչը աղմկոտ է դարձնում ընթերցումները։ Կարճ շարք դիտելը և
    *ամենից հաճախ* տեսնված դասը վերցնելը (հավասարությունը լուծվում է ընդհանուր
    վստահությամբ) շատ ավելի կայուն է։

    Վերադարձնում է ``(class_name, avg_confidence, votes)`` հաղթող դասի համար,
    որտեղ ``votes``-ը ``{class_name: count}`` բառարան է, կամ ``None``, եթե ոչ մի
    դաս չի հայտնվել կադրերի առնվազն կեսում (մեր «սխալի» դեպքը — չափազանց անվստահ)։
    """
    counts: Counter = Counter()          # how many frames saw each class
                                         # քանի՞ կադր տեսավ ամեն դասը
    conf_sum: dict = defaultdict(float)  # total confidence per class (for tie-breaks)
                                         # ընդհանուր վստահությունը ամեն դասի համար (հավասարությունների լուծման)

    for _ in range(frames):
        result = detect_hand(model, camera.read(), conf_threshold, device)
        if result is not None:
            name, conf = result
            counts[name] += 1
            conf_sum[name] += conf

    if not counts:
        return None  # nothing confident in any frame
                     # ոչ մի կադրում ոչինչ վստահ չէր

    # Winner = most frequent class; if two tie on count, the higher total
    # confidence wins.
    # Հաղթող = ամենահաճախ հանդիպող դասը. եթե երկուսը հավասար են քանակով,
    # հաղթում է ավելի բարձր ընդհանուր վստահությունը։
    winner = max(counts, key=lambda name: (counts[name], conf_sum[name]))

    # Require the winner in at least half the frames, otherwise it is too shaky
    # to trust (e.g. 2 votes Rock, 2 votes Scissors, 1 nothing -> error).
    # Պահանջում ենք, որ հաղթողը լինի կադրերի առնվազն կեսում, այլապես չափազանց
    # անկայուն է վստահելու համար (օր.՝ 2 քվե Քար, 2 քվե Մկրատ, 1 ոչինչ -> սխալ)։
    if counts[winner] < (frames + 1) // 2:
        return None

    avg_confidence = conf_sum[winner] / counts[winner]
    return winner, avg_confidence, dict(counts)


# ===========================================================================
# The game loop
# Խաղի ցիկլը
# ===========================================================================
def play_round(model, camera, conf_threshold, device, frames) -> str | None:
    """Play exactly one round. Returns "win"/"lose"/"tie", or None on error.

    Խաղում է ուղիղ մեկ ռաունդ։ Վերադարձնում է "win"/"lose"/"tie", կամ None՝ սխալի դեպքում։
    """
    # A short countdown gives the player time to form their shape.
    # Կարճ հետհաշվարկը խաղացողին ժամանակ է տալիս ձևավորելու իր ձեռքի պատկերը։
    for n in (3, 2, 1):
        print(f"  {n}...", end=" ", flush=True)
        time.sleep(0.8)
    print("shoot!")

    # Capture a short burst and vote, instead of trusting one (often noisy) frame.
    # Վերցնում ենք կարճ շարք և քվեարկում, փոխանակ վստահելու մեկ (հաճախ աղմկոտ) կադրի։
    detection = detect_hand_burst(model, camera, frames, conf_threshold, device)

    # No confident detection -> show the "error" outcome and let them retry.
    # Վստահ հայտնաբերում չկա -> ցույց ենք տալիս «սխալի» արդյունքը և թույլ տալիս կրկնել։
    if detection is None:
        print("  🤖 error: I couldn't see a clear hand. Try again!")
        return None

    class_name, confidence, votes = detection
    player_move = detection_to_move(class_name)      # e.g. "Rock" -> "rock"
    computer_move = random_move()
    outcome = decide_winner(player_move, computer_move)

    # Friendly summary of the round.
    # Ռաունդի ընկերական ամփոփումը։
    print(f"  👋 You played : {player_move}  "
          f"(confidence {confidence:.0%}, seen in {votes[class_name]}/{frames} frames)")
    print(f"  🤖 I played   : {computer_move}")
    verdict = {"win": "You win! 🎉", "lose": "I win! 🤖", "tie": "It's a tie 🤝"}[outcome]
    print(f"  >> {verdict}")
    return outcome


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Rock-Paper-Scissors against the YOLO11 model.")
    parser.add_argument("--weights", default=None,
                        help="Model to play with: a best.pt file or an NCNN folder "
                             "(default: newest NCNN export if present, else best.pt).")
    parser.add_argument("--camera", default="picamera2", choices=["picamera2", "opencv"],
                        help="Camera backend (default: %(default)s; use opencv to test on a laptop).")
    parser.add_argument("--conf", type=float, default=config.CONF_THRESHOLD,
                        help="Minimum confidence to accept a detection (default: %(default)s).")
    parser.add_argument("--frames", type=int, default=config.PLAY_FRAMES,
                        help="Frames to capture and vote over each round (default: %(default)s).")
    parser.add_argument("--rounds", type=int, default=0,
                        help="Number of rounds to play; 0 means keep playing until you quit.")
    parser.add_argument("--device", default=config.get_device(),
                        help="Compute device (default: auto-detected; the Pi uses cpu).")
    return parser.parse_args()


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

    # Running tally of results from the player's point of view.
    # Արդյունքների ընթացիկ հաշիվը՝ խաղացողի տեսանկյունից։
    score = {"win": 0, "lose": 0, "tie": 0}

    print("\n=== Rock-Paper-Scissors vs. the computer ===")
    print("Press Enter to play a round, or type 'q' then Enter to quit.\n")

    try:
        round_number = 0
        while True:
            # Stop after a fixed number of rounds if --rounds was given.
            # Կանգ ենք առնում որոշակի քանակի ռաունդներից հետո, եթե --rounds-ը տրված է։
            if args.rounds and round_number >= args.rounds:
                break

            command = input(f"Round {round_number + 1} — ready? [Enter / q] ").strip().lower()
            if command == "q":
                break

            outcome = play_round(model, camera, args.conf, args.device, args.frames)
            if outcome is None:
                # Errors don't count as a round; let the player try again.
                # Սխալները ռաունդ չեն համարվում. թույլ ենք տալիս խաղացողին նորից փորձել։
                continue

            score[outcome] += 1
            round_number += 1
            print(f"  Score so far — you {score['win']} : {score['lose']} me  "
                  f"({score['tie']} ties)\n")
    except KeyboardInterrupt:
        # Ctrl-C is a perfectly normal way to stop the game.
        # Ctrl-C-ն խաղը դադարեցնելու լիովին նորմալ ձև է։
        print("\nInterrupted.")
    finally:
        # Always release the camera, even if something went wrong.
        # Միշտ ազատում ենք տեսախցիկը, նույնիսկ եթե ինչ-որ բան սխալ է գնացել։
        camera.close()

    print("\nThanks for playing!")
    print(f"Final score — you {score['win']} : {score['lose']} me ({score['tie']} ties)")


if __name__ == "__main__":
    main()
