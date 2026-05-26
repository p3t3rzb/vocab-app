"""CLI entry point for training: ``uv run python -m src.model --db <path>``."""
import argparse

from src.database import init_db
from src.model.config import TrainConfig
from src.model.training import train


def _parse_args() -> argparse.Namespace:
    """Parse CLI flags, falling back to :class:`TrainConfig` defaults."""
    defaults = TrainConfig()
    p = argparse.ArgumentParser(description="Train RecallLSTM on repetition history")
    p.add_argument("--db", required=True, help="Path to the SQLite database, e.g. storage/french.db")
    p.add_argument("--epochs", type=int, default=defaults.epochs)
    p.add_argument("--hidden-size", type=int, default=defaults.hidden_size)
    p.add_argument("--num-layers", type=int, default=defaults.num_layers)
    p.add_argument("--dropout", type=float, default=defaults.dropout)
    p.add_argument("--lr", type=float, default=defaults.lr)
    p.add_argument("--batch-size", type=int, default=defaults.batch_size)
    p.add_argument("--val-split", type=float, default=defaults.val_split)
    p.add_argument("--seed", type=int, default=defaults.seed)
    return p.parse_args()


def main() -> None:
    """Initialise the requested database and kick off training."""
    args = _parse_args()
    db_url = f"sqlite:///{args.db}" if not args.db.startswith("sqlite") else args.db
    cfg = TrainConfig(
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
    )
    init_db(db_url, "", "")
    train(config=cfg)


if __name__ == "__main__":
    main()
