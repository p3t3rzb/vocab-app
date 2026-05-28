# vocab-repetition

A spaced-repetition vocabulary app. You build a list of words in two languages,
practice them in both directions, and the app learns when to show each word again
so you review it right before you'd forget it.

## Requirements

- [uv](https://docs.astral.sh/uv/) (handles Python and all dependencies)

## Running the app

```bash
uv sync       # install dependencies (first time only)
uv run python -m src.main
```

That opens the desktop window. Everything below is done from inside the app.

## Using the app

### 1. Pick or create a database

Each language pair lives in its own database. On the first screen you'll see a list
of existing databases.

- **Open** an existing one to start working with it.
- **New** creates a fresh database: enter the source and target language names
  (e.g. *French* and *Polish*) and click Create. You start with an empty word list.

### 2. Manage your words

The word list shows every word, its translation, and when each direction is next
due for review. Use the search box to filter, and the toolbar to:

- **Add** a new word (source text + translation).
- **Edit** or **Delete** the selected word.
- Click a word to see its full **practice history**.

### 3. Practice

Click **Practice** to start a review session. The app builds a queue of the words
that are due, plus any you haven't learned yet, and quizzes you in both directions:

1. The word is shown in one language.
2. Press **↓** to reveal the translation.
3. Press **→** if you remembered it, **←** if you didn't.
4. The app records the result, schedules when you'll see the word next, and you
   move on to the next card.

Words you just failed come back later in the same session. Reviews always come
before brand-new words.

### 4. Train the recall model

The scheduling gets smarter once you've built up some practice history. Open a
database and click **Train Model**, set the number of training passes (epochs),
and click Train. A live plot shows progress; the app trains in the background so
the window stays responsive. When training finishes, every word's next-review
time is recalculated automatically.

Until a model is trained, you can still practice — the app just won't predict
review times yet.

### 5. Settings

From the home screen, click **Settings** to adjust:

- **Recall threshold** — how confident the app must be that you'd remember a word
  before it considers it due. Lower = you review more often.
- **Max interval** — a cap on how far into the future a review can be scheduled.
- **Appearance** — Light, Dark, or System.

Saving a changed threshold or interval re-schedules all your words across every
database that has a trained model.
