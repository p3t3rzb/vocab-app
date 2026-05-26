# vocab-repetition

A spaced-repetition vocabulary app with an LSTM recall predictor.

## Stack

- **GUI** — customtkinter desktop app (database browser, word list, word detail, add/edit/delete)
- **Database** — SQLite via SQLAlchemy 2.0; one `.db` file per language pair
- **Model** — 2-layer LSTM trained on repetition history; predicts P(remembered) given time since last practice and previous recall outcome

## Quick start

```bash
# Install dependencies
uv sync

# Run the GUI
TCL_LIBRARY="$HOME/.local/share/uv/python/cpython-3.13.5-macos-aarch64-none/lib/tcl8.6" \
TK_LIBRARY="$HOME/.local/share/uv/python/cpython-3.13.5-macos-aarch64-none/lib/tk8.6" \
uv run python -m src.main

# Train the recall model
uv run python -m src.model.train --db storage/french.db --epochs 50
```

## Project layout

```
src/
  main.py          — GUI entry point
  settings.py      — AppSettings dataclass + load/save (storage/settings.json)
  database/        — ORM layer (models, repositories, session)
  gui/             — customtkinter screens
    train_screen.py    — in-app training screen with live loss plot
    settings_screen.py — global settings (threshold, max interval, appearance)
  model/           — LSTM recall predictor
    config.py      — TrainConfig dataclass (single source of hyperparameter defaults)
    lstm.py        — RecallLSTM architecture
    dataset.py     — DB → training sequences pipeline
    train.py       — training loop + CLI
storage/
  french.db        — 8,790 French↔Polish words, 255,455 repetition events
  settings.json    — user-configurable global settings (created on first save)
  models/          — saved model checkpoints (<source>_<target>.pt)
```

See `CLAUDE.md` for full documentation.
