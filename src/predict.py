"""
Predict the next repetition time for a word using the trained RecallLSTM.

Usage:
    uv run python -m src.predict --db storage/french_polish.db --word-id 42
    uv run python -m src.predict --db storage/french_polish.db --word-id 42 --direction reverse
    uv run python -m src.predict --db storage/french_polish.db --word salut
    uv run python -m src.predict --db storage/french_polish.db --word salut --history
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

from src.database import (
    Direction,
    LanguagePairRepository,
    RepetitionRepository,
    WordRepository,
    get_session,
    init_db,
)
from src.model import Predictor, load_model
from src.model.config import PredictConfig


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as a compact human string (e.g. ``"2d 4h 13m"``)."""
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes = seconds // 60
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}m")
    return " ".join(parts)


def _predict_for_direction(
    predictor: Predictor,
    word_id: int,
    source_text: str,
    target_text: str,
    direction: Direction,
    show_history: bool,
) -> None:
    """Print the next-repetition prediction (and optional history) for one direction."""
    with get_session() as session:
        reps = RepetitionRepository(session).get_for_word(word_id, direction)

    if not reps:
        return

    dir_label = "FORWARD" if direction == Direction.FORWARD else "REVERSE"
    arrow = f"{source_text} → {target_text}" if direction == Direction.FORWARD else f"{target_text} → {source_text}"
    print(f"\n{'─' * 60}")
    print(f"  {arrow}  [{dir_label}]")
    print(f"{'─' * 60}")
    print(f"  Repetitions:   {len(reps)}")

    last_rep = reps[-1]
    last_dt = datetime.fromtimestamp(last_rep.practiced_at)
    elapsed = time.time() - last_rep.practiced_at
    print(f"  Last practiced: {last_dt.strftime('%Y-%m-%d %H:%M')}  ({_fmt_duration(elapsed)} ago)")

    p_now = predictor.recall_probability(reps, elapsed)
    print(f"  P(recall now):  {p_now:.3f}")

    delta = predictor.next_repetition_delta(reps)
    scheduled_ts = last_rep.practiced_at + delta
    scheduled_dt = datetime.fromtimestamp(scheduled_ts)
    print(f"\n  Next repetition in: {_fmt_duration(delta)}")
    print(f"  Scheduled for:      {scheduled_dt.strftime('%Y-%m-%d %H:%M')}")

    if show_history and len(reps) >= 3:
        cfg = predictor.config
        print(f"\n  History (threshold={cfg.recall_threshold}):")
        print(f"  {'#':>3}  {'Predicted':>12}  {'Actual':>12}  {'P@actual':>10}  {'Remembered'}")
        print(f"  {'─'*3}  {'─'*12}  {'─'*12}  {'─'*10}  {'─'*10}")
        for i in range(1, len(reps) - 1):
            context = reps[:i]
            pred_delta = predictor.next_repetition_delta(context)
            actual_delta = reps[i].practiced_at - reps[i - 1].practiced_at
            p_actual = predictor.recall_probability(context, actual_delta)
            remembered = "yes" if reps[i].remembered else "no"
            print(
                f"  {i:>3}  {_fmt_duration(pred_delta):>12}  "
                f"{_fmt_duration(actual_delta):>12}  "
                f"{p_actual:>10.3f}  {remembered}"
            )


def _parse_args() -> argparse.Namespace:
    """Parse CLI flags for the predict script."""
    p = argparse.ArgumentParser(description="Predict next repetition time for a vocabulary word")
    p.add_argument("--db", required=True, help="Path to the SQLite database, e.g. storage/french_polish.db")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--word-id", type=int, help="Word ID")
    group.add_argument("--word", type=str, help="Partial match on source or target text")
    p.add_argument(
        "--direction",
        choices=["forward", "reverse", "both"],
        default="both",
        help="Which direction to predict (default: both)",
    )
    p.add_argument("--history", action="store_true", help="Show historical interval comparison")
    return p.parse_args()


def main() -> None:
    """Entry point: resolve the word(s) to predict, load the model, print results."""
    args = _parse_args()

    db_path = args.db
    db_url = f"sqlite:///{db_path}" if not db_path.startswith("sqlite") else db_path
    init_db(db_url, "", "")

    with get_session() as session:
        lp = LanguagePairRepository(session).get()
        if lp is None:
            print("Error: could not read language pair from database.")
            return
        source_lang = lp.source_language
        target_lang = lp.target_language

        ckpt_path = Path("storage") / "models" / f"{source_lang.lower()}_{target_lang.lower()}.pt"
        if not ckpt_path.exists():
            print(f"Error: no model checkpoint found at {ckpt_path}")
            print("Run 'uv run python -m src.model --db <db>' to train first.")
            return

        words_repo = WordRepository(session)

        if args.word_id is not None:
            word = words_repo.get_by_id(args.word_id)
            if word is None:
                print(f"Error: no word with id {args.word_id}")
                return
            matches = [word]
        else:
            query = args.word.lower()
            all_words = words_repo.get_all()
            matches = [
                w for w in all_words
                if query in w.source_text.lower() or query in w.target_text.lower()
            ]
            if not matches:
                print(f"No words matching '{args.word}'")
                return
            if len(matches) > 5:
                print(f"Found {len(matches)} matches — showing first 5. Use --word-id for a specific word.")
                matches = matches[:5]

    model = load_model(ckpt_path)
    predictor = Predictor(model, PredictConfig())

    directions: list[Direction] = []
    if args.direction in ("forward", "both"):
        directions.append(Direction.FORWARD)
    if args.direction in ("reverse", "both"):
        directions.append(Direction.REVERSE)

    for word in matches:
        print(f"\nWord #{word.id}: {word.source_text} / {word.target_text}")
        for direction in directions:
            _predict_for_direction(
                predictor,
                word.id,
                word.source_text,
                word.target_text,
                direction,
                args.history,
            )


if __name__ == "__main__":
    main()
