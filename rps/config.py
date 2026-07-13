"""
config.py
=========

A single, central place for the settings that the other scripts share.
Մեկ կենտրոնական վայր՝ այն կարգավորումների համար, որ մյուս սկրիպտները կիսում են։

Keeping constants here (instead of copy-pasting them into every script) means:
  * there is ONE obvious place to change a value, and
  * every script automatically stays consistent with the others.
Հաստատունները այստեղ պահելը (փոխանակ ամեն սկրիպտում կրկնօրինակելու) նշանակում է՝
  * կա ՄԵԿ ակնհայտ տեղ՝ արժեքը փոխելու համար, և
  * ամեն սկրիպտ ինքնաշխատ համաձայնեցված է մնում մյուսների հետ։

Nothing in this file does any real work; it only *describes* things.
Read this file first when you want to understand the project.
Այս ֆայլը իրական աշխատանք չի կատարում. այն միայն *նկարագրում* է բաները։
Կարդացե՛ք այս ֆայլը առաջինը, երբ ուզում եք հասկանալ նախագիծը։
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# 1. The classes our model knows about
#    Դասերը (classes), որ մեր մոդելը ճանաչում է
# ---------------------------------------------------------------------------
# IMPORTANT: the ORDER matters. YOLO labels each object with a number (0, 1, 2)
# and the dataset's `data.yaml` defines which number means which class:
# ԿԱՐԵՎՈՐ Է՝ ՀԵՐԹԱԿԱՆՈՒԹՅՈՒՆԸ կարևոր է։ YOLO-ն ամեն օբյեկտին տալիս է թիվ (0, 1, 2),
# իսկ dataset-ի `data.yaml`-ը սահմանում է, թե որ թիվը որ դասն է նշանակում.
#
#       0 -> Paper    (Թուղթ)
#       1 -> Rock     (Քար)
#       2 -> Scissors (Մկրատ)
#
# We copy that exact order here so that `CLASS_NAMES[class_id]` always gives the
# correct human-readable name. Do NOT alphabetise this list.
# Մենք պատճենում ենք ճիշտ նույն հերթականությունն այստեղ, որպեսզի `CLASS_NAMES[class_id]`-ը
# միշտ տա ճիշտ, մարդու համար ընթեռնելի անունը։ ՄԻ՛ դասավորեք այս ցանկն այբբենական կարգով։
CLASS_NAMES: list[str] = ["Paper", "Rock", "Scissors"]


# ---------------------------------------------------------------------------
# 2. Where things live on disk
#    Որտեղ են ֆայլերը գտնվում սկավառակի վրա
# ---------------------------------------------------------------------------
# `Path(__file__).resolve().parent` is the folder that contains THIS file,
# i.e. the project root. Building paths relative to it means the scripts work
# no matter what directory you run them from.
# `Path(__file__).resolve().parent`-ը այն թղթապանակն է, որը պարունակում է ԱՅՍ ֆայլը,
# այսինքն՝ նախագծի արմատը (root)։ Ուղիները դրա նկատմամբ կառուցելը նշանակում է, որ
# սկրիպտներն աշխատում են՝ անկախ նրանից, թե որ թղթապանակից եք դրանք գործարկում։
PROJECT_ROOT: Path = Path(__file__).resolve().parent

# The dataset zip we were given (downloaded from Roboflow).
# Տվյալների հավաքածուի (dataset) zip ֆայլը, որ մեզ տրվել է (ներբեռնված Roboflow-ից)։
DATASET_ZIP: Path = Path.home() / "Downloads" / "rock-paper-scissors.v14i.yolov11.zip"

# Where `prepare_dataset.py` will unpack the images and labels.
# Որտեղ `prepare_dataset.py`-ը կբացի (unpack) նկարներն ու պիտակները (labels)։
DATASET_DIR: Path = PROJECT_ROOT / "dataset"

# The dataset description file that YOLO reads to find the train/val/test images.
# Dataset-ի նկարագրության ֆայլը, որ YOLO-ն կարդում է՝ գտնելու train/val/test նկարները։
DATA_YAML: Path = DATASET_DIR / "data.yaml"

# Ultralytics writes training runs (weights, plots, metrics) under here by default.
# Ultralytics-ը լռելյայն այստեղ է գրում ուսուցման արդյունքները (կշիռներ, գրաֆիկներ, չափումներ)։
RUNS_DIR: Path = PROJECT_ROOT / "runs"


# ---------------------------------------------------------------------------
# 3. Sensible defaults for training / inference
#    Խելամիտ լռելյայն արժեքներ ուսուցման / կանխատեսման համար
# ---------------------------------------------------------------------------
# These are starting points tuned for teaching, not competition. Every script
# also exposes them as command-line flags so students can experiment.
# Սրանք ելակետային արժեքներ են՝ հարմարեցված ուսուցման, ոչ թե մրցակցության համար։ Ամեն սկրիպտ
# դրանք նաև ներկայացնում է որպես հրամանային տողի դրոշներ (flags), որպեսզի ուսանողները փորձարկեն։

# The pretrained model we start from. "yolo11n" is the *nano* size: the smallest
# and fastest YOLO11, which is exactly what we want for real-time detection on a
# Raspberry Pi 5. The first time you train, Ultralytics downloads this file.
# Նախապես ուսուցանված մոդելը, որից սկսում ենք։ "yolo11n"-ը *nano* չափսն է՝ ամենափոքր և
# ամենաարագ YOLO11-ը, որը հենց այն է, ինչ մեզ պետք է Raspberry Pi 5-ի վրա իրական ժամանակում
# հայտնաբերման (detection) համար։ Առաջին ուսուցման ժամանակ Ultralytics-ը ներբեռնում է այս ֆայլը։
BASE_MODEL: str = "yolo11n.pt"

# All images in this dataset are 640x640, so we train and infer at that size.
# Այս dataset-ի բոլոր նկարները 640x640 են, ուստի ուսուցումն ու կանխատեսումն անում ենք այդ չափսով։
IMG_SIZE: int = 640

# How many times the model sees the whole training set. More epochs -> more
# learning (up to a point), but slower. 60 is a reasonable teaching default.
# Քանի անգամ է մոդելը տեսնում ամբողջ ուսուցման հավաքածուն։ Ավելի շատ epoch -> ավելի շատ
# ուսուցում (մինչև որոշ սահման), բայց ավելի դանդաղ։ 60-ը խելամիտ լռելյայն արժեք է ուսուցման համար։
EPOCHS: int = 60

# How many images the GPU processes at once. -1 lets Ultralytics pick a value
# that fits in memory automatically — handy on the Mac's shared memory.
# Քանի նկար է GPU-ն մշակում միաժամանակ։ -1-ը թույլ է տալիս Ultralytics-ին ինքնաշխատ ընտրել
# հիշողության մեջ տեղավորվող արժեք — հարմար է Mac-ի ընդհանուր (shared) հիշողության դեպքում։
BATCH: int = -1

# During the game, a detection must be at least this confident (0-1) to count.
# Anything below this is treated as "no hand detected" -> the game shows "error".
# Խաղի ընթացքում հայտնաբերումը պետք է լինի առնվազն այսքան վստահ (0-1)՝ հաշվվելու համար։
# Սրանից ցածր ամեն ինչ համարվում է «ձեռք չի հայտնաբերվել» -> խաղը ցույց է տալիս «error»։
CONF_THRESHOLD: float = 0.50

# A single frame can be blurry or caught mid-move, giving noisy results. At play
# time we instead capture this many frames in quick succession and take the class
# seen most often (a majority vote). 5 frames is ~0.85 s with the NCNN model on a
# Raspberry Pi 5.
# Մեկ կադրը (frame) կարող է լղոզված լինել կամ բռնված շարժման կեսին՝ տալով աղմկոտ արդյունքներ։
# Խաղի ժամանակ փոխարենը գրանցում ենք այսքան կադր՝ արագ հաջորդականությամբ, և վերցնում ամենից շատ
# հանդիպող դասը (մեծամասնության քվեարկություն)։ 5 կադրը ~0.85 վրկ է NCNN մոդելով Raspberry Pi 5-ի վրա։
PLAY_FRAMES: int = 5


# ---------------------------------------------------------------------------
# 4. Picking the hardware to run on
#    Ընտրել սարքավորումը (hardware), որի վրա աշխատել
# ---------------------------------------------------------------------------
def get_device() -> str:
    """Return the best available compute device for PyTorch as a short string.
    Վերադարձնում է PyTorch-ի համար լավագույն հասանելի հաշվարկային սարքը՝ որպես կարճ տող։

    Ultralytics understands these strings directly:
    Ultralytics-ն ուղղակիորեն հասկանում է այս տողերը՝
        "mps"  -> Apple Silicon GPU (Metal) — what we use on this Mac
                  Apple Silicon GPU (Metal) — ինչ որ մենք օգտագործում ենք այս Mac-ի վրա
        "cuda" -> NVIDIA GPU (e.g. a cloud machine or Colab)
                  NVIDIA GPU (օրինակ՝ ամպային մեքենա կամ Colab)
        "cpu"  -> no GPU; works everywhere but is slowest
                  առանց GPU-ի. աշխատում է ամենուր, բայց ամենադանդաղն է

    We import torch *inside* the function so that simply importing `config`
    (for example from the pure-Python game logic) does not require torch to be
    installed.
    Մենք ներմուծում ենք torch-ը ֆունկցիայի *ներսում*, որպեսզի պարզապես `config`-ը ներմուծելը
    (օրինակ՝ մաքուր Python-ով գրված խաղի տրամաբանությունից) չպահանջի, որ torch-ը տեղադրված լինի։
    """
    try:
        import torch
    except ImportError:
        # torch isn't installed (e.g. on a minimal setup) — fall back to CPU.
        # torch-ը տեղադրված չէ (օրինակ՝ նվազագույն կարգավորման դեպքում) — անցնում ենք CPU-ի։
        return "cpu"

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# 5. Finding the weights produced by training
#    Գտնել ուսուցման արդյունքում ստացված կշիռները (weights)
# ---------------------------------------------------------------------------
# Training writes its best model to runs/detect/rps_train/weights/best.pt
# (or rps_train2, rps_train3, ... if you train more than once). This helper
# returns the newest one so evaluate.py / export.py / play.py can default to it.
# Ուսուցումը գրում է իր լավագույն մոդելը runs/detect/rps_train/weights/best.pt հասցեում
# (կամ rps_train2, rps_train3, ... եթե ուսուցանում եք մեկից ավելի անգամ)։ Այս օժանդակ ֆունկցիան
# վերադարձնում է ամենանորը, որպեսզի evaluate.py / export.py / play.py-ն այն օգտագործեն լռելյայն։
def default_weights() -> Path:
    """Return the path to the most recently trained ``best.pt`` (newest run).
    Վերադարձնում է ամենավերջին ուսուցանված ``best.pt``-ի ուղին (ամենանոր run-ը)։

    Falls back to the canonical first-run location if nothing is found yet, so
    error messages point somewhere sensible.
    Եթե դեռ ոչինչ չի գտնվել, վերադառնում է առաջին run-ի ստանդարտ հասցեին, որպեսզի
    սխալի հաղորդագրությունները մատնանշեն խելամիտ տեղ։
    """
    detect_dir = RUNS_DIR / "detect"
    candidates = sorted(detect_dir.glob("rps_train*/weights/best.pt"),
                        key=lambda p: p.stat().st_mtime)
    if candidates:
        return candidates[-1]  # the last one is the most recently modified
                               # վերջինը ամենավերջին փոփոխվածն է
    return detect_dir / "rps_train" / "weights" / "best.pt"


def default_play_weights() -> Path:
    """Return the best weights to PLAY with, preferring the fast NCNN export.
    Վերադարձնում է ԽԱՂԱԼՈՒ համար լավագույն կշիռները՝ նախապատվությունը տալով արագ NCNN արտահանմանը։

    `export.py` converts ``best.pt`` into a ``best_ncnn_model`` folder that runs
    roughly 5x faster on the Pi's CPU, so at play time we use the newest NCNN
    export if one exists. If you have not exported yet, we fall back to the newest
    ``best.pt`` so the game still works.
    `export.py`-ն ``best.pt``-ն վերածում է ``best_ncnn_model`` թղթապանակի, որը Pi-ի CPU-ի վրա
    աշխատում է մոտ 5 անգամ ավելի արագ, ուստի խաղի ժամանակ օգտագործում ենք ամենանոր NCNN
    արտահանումը, եթե այդպիսին կա։ Եթե դեռ չեք արտահանել, վերադառնում ենք ամենանոր ``best.pt``-ին,
    որպեսզի խաղը դեռ աշխատի։
    """
    detect_dir = RUNS_DIR / "detect"
    ncnn = sorted(detect_dir.glob("rps_train*/weights/best_ncnn_model"),
                  key=lambda p: p.stat().st_mtime)
    if ncnn:
        return ncnn[-1]
    return default_weights()  # no NCNN export yet -> use the PyTorch weights
                              # դեռ NCNN արտահանում չկա -> օգտագործում ենք PyTorch կշիռները
