"""
game_logic.py
=============

The *rules* of Rock-Paper-Scissors, written in plain Python with no machine
learning and no camera code at all.
Քար-Թուղթ-Մկրատ խաղի *կանոնները*՝ գրված պարզ Python-ով, առանց
մեքենայական ուսուցման և առանց տեսախցիկի կոդի։

Why keep this separate?
  * It is the easiest part to read and to teach.
  * It has no heavy dependencies, so it runs instantly and is trivial to test.
  * The play loop (`play.py`) imports these functions, so the game rules live
    in exactly one place.
Ինչու՞ պահել սա առանձին։
  * Սա ամենահեշտ ընթերցվող և ուսուցանվող մասն է։
  * Այն ծանր կախվածություններ չունի, ուստի գործարկվում է ակնթարթորեն և
    շատ հեշտ է թեստավորվում։
  * Խաղի ցիկլը (`play.py`) ներմուծում է այս ֆունկցիաները, ուստի խաղի
    կանոնները ապրում են ուղիղ մեկ տեղում։

The three moves are represented as the lowercase strings "rock", "paper" and
"scissors". We deliberately reuse plain strings (rather than a fancy class) to
keep the code approachable for students.
Երեք քայլերը ներկայացված են որպես փոքրատառ տողեր՝ "rock", "paper" և
"scissors"։ Մենք միտումնավոր օգտագործում ենք պարզ տողեր (այլ ոչ թե բարդ
class), որպեսզի կոդը մատչելի լինի աշակերտների համար։
"""

from __future__ import annotations

import random

from config import CLASS_NAMES


# ---------------------------------------------------------------------------
# The three legal moves
# Երեք թույլատրելի քայլերը
# ---------------------------------------------------------------------------
ROCK = "rock"
PAPER = "paper"
SCISSORS = "scissors"

# A list is handy when we want to pick a random move for the computer.
# Ցուցակը հարմար է, երբ ուզում ենք համակարգչի համար պատահական քայլ ընտրել։
MOVES: list[str] = [ROCK, PAPER, SCISSORS]


# ---------------------------------------------------------------------------
# Who beats whom?
# Ո՞վ ում է հաղթում։
# ---------------------------------------------------------------------------
# Read this as: "the KEY move beats the VALUE move".
#   rock     beats scissors   (rock smashes scissors)
#   paper    beats rock       (paper covers rock)
#   scissors beats paper      (scissors cut paper)
# Կարդացեք սա այսպես՝ «ԲԱՆԱԼԻ քայլը հաղթում է ԱՐԺԵՔ քայլին»։
#   քար    հաղթում է մկրատին    (քարը ջարդում է մկրատը)
#   թուղթ  հաղթում է քարին      (թուղթը ծածկում է քարը)
#   մկրատ  հաղթում է թղթին      (մկրատը կտրում է թուղթը)
BEATS: dict[str, str] = {
    ROCK: SCISSORS,
    PAPER: ROCK,
    SCISSORS: PAPER,
}


def random_move() -> str:
    """Return one of "rock", "paper", "scissors" at random — the computer's hand.
    Վերադարձնում է "rock", "paper" կամ "scissors"-ից մեկը պատահականորեն՝
    համակարգչի ձեռքը։
    """
    return random.choice(MOVES)


def detection_to_move(class_name: str | None) -> str | None:
    """Translate a model's class name (e.g. "Rock") into a game move ("rock").
    Թարգմանում է մոդելի class-ի անունը (օր․՝ "Rock") խաղի քայլի ("rock")։

    The model's class names are capitalised ("Rock", "Paper", "Scissors"); our
    moves are lowercase. This helper bridges the two.
    Մոդելի class-ի անունները մեծատառ են ("Rock", "Paper", "Scissors"), իսկ
    մեր քայլերը՝ փոքրատառ։ Այս օգնական ֆունկցիան կապում է երկուսը։

    Returns the matching move, or ``None`` if there was no usable detection
    (``class_name`` is ``None``). ``None`` is what the play loop turns into the
    on-screen "error" message.
    Վերադարձնում է համապատասխան քայլը, կամ ``None``, եթե օգտագործելի
    հայտնաբերում չկար (``class_name``-ը ``None`` է)։ ``None``-ը այն է, ինչ խաղի
    ցիկլը վերածում է էկրանի "error" հաղորդագրության։
    """
    if class_name is None:
        return None

    move = class_name.lower()
    # Guard against an unexpected label so a typo in the dataset can't crash the game.
    # Պաշտպանվում ենք անսպասելի պիտակից, որպեսզի տվյալների բազայի վրիպակը
    # չկարողանա խաղը խափանել։
    return move if move in MOVES else None


def decide_winner(player: str, computer: str) -> str:
    """Compare two moves and say how it went *from the player's point of view*.
    Համեմատում է երկու քայլերը և ասում, թե ինչպես ընթացավ խաղը՝
    *խաղացողի տեսանկյունից*։

    Returns one of:
        "tie"  -> same move on both sides
        "win"  -> the player's move beats the computer's
        "lose" -> the computer's move beats the player's
    Վերադարձնում է հետևյալներից մեկը՝
        "tie"  -> միևնույն քայլը երկու կողմից (ոչ-ոքի)
        "win"  -> խաղացողի քայլը հաղթում է համակարգչի քայլին
        "lose" -> համակարգչի քայլը հաղթում է խաղացողի քայլին
    """
    if player == computer:
        return "tie"
    # BEATS[player] is the move that `player` defeats. If that equals the
    # computer's move, the player wins; otherwise the player loses.
    # BEATS[player]-ը այն քայլն է, որին `player`-ը հաղթում է։ Եթե դա հավասար
    # է համակարգչի քայլին՝ խաղացողը հաղթում է, հակառակ դեպքում՝ պարտվում է։
    if BEATS[player] == computer:
        return "win"
    return "lose"


# ---------------------------------------------------------------------------
# Self-check
# Ինքնաստուգում
# ---------------------------------------------------------------------------
# Running `python game_logic.py` directly executes the block below. It is a tiny
# built-in test so students can confirm the rules behave as expected without any
# extra tools. It does nothing when the module is imported by another script.
# `python game_logic.py`-ն ուղղակիորեն գործարկելը կատարում է ստորև բլոկը։
# Սա փոքրիկ ներկառուցված թեստ է, որպեսզի աշակերտները կարողանան հաստատել, որ
# կանոնները աշխատում են ինչպես սպասվում է՝ առանց լրացուցիչ գործիքների։ Այն ոչինչ
# չի անում, երբ մոդուլը ներմուծվում է մեկ այլ սկրիպտի կողմից։
if __name__ == "__main__":
    # Sanity check 1: every class name the model can produce maps to a real move.
    # Ստուգում 1՝ մոդելի արտադրած ամեն class-ի անուն համապատասխանում է
    # իրական քայլի։
    for name in CLASS_NAMES:
        assert detection_to_move(name) in MOVES, name
    # A missing detection becomes None ("error").
    # Բացակայող հայտնաբերումը դառնում է None ("error")։
    assert detection_to_move(None) is None

    # Sanity check 2: the outcome table is correct for all 9 combinations.
    # Ստուգում 2՝ արդյունքների աղյուսակը ճիշտ է բոլոր 9 համակցությունների համար։
    expected = {
        (ROCK, ROCK): "tie",        (ROCK, PAPER): "lose",     (ROCK, SCISSORS): "win",
        (PAPER, ROCK): "win",       (PAPER, PAPER): "tie",     (PAPER, SCISSORS): "lose",
        (SCISSORS, ROCK): "lose",   (SCISSORS, PAPER): "win",  (SCISSORS, SCISSORS): "tie",
    }
    for (p, c), want in expected.items():
        got = decide_winner(p, c)
        assert got == want, f"{p} vs {c}: expected {want}, got {got}"

    print("All game-logic checks passed ✅")
    print("Example round -> player:rock  computer:", (demo := random_move()),
          " result:", decide_winner(ROCK, demo))
