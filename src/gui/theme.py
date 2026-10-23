"""Central theming and configuration constants for the GUI.

Every magic number, font size, color, padding tuple, window geometry,
poll interval, validation limit, and default value used by the GUI lives
here. The rest of the GUI modules import these classes as namespaces.

The font kwargs are stored as plain dicts so they can be passed to
``customtkinter.CTkFont(**Fonts.HEADER)`` at widget-creation time —
``CTkFont`` requires a running Tk root, so we cannot pre-instantiate it.
"""

from __future__ import annotations

from pathlib import Path

from src import paths as _paths


class Fonts:
    """Keyword args for :class:`customtkinter.CTkFont`."""

    TITLE = {"size": 22, "weight": "bold"}
    HEADER = {"size": 18, "weight": "bold"}
    SECTION = {"size": 14, "weight": "bold"}
    BODY = {"size": 14}
    BODY_BOLD = {"size": 14, "weight": "bold"}
    SMALL = {"size": 13}
    PROMPT = {"size": 42, "weight": "bold"}
    # The prompt once the answer is revealed — demoted so the answer leads.
    PROMPT_REVEALED = {"size": 28, "weight": "bold"}
    ANSWER = {"size": 42, "weight": "bold"}
    # Full-card messages (session done, offer to keep practicing).
    MESSAGE = {"size": 22, "weight": "bold"}
    CHIP = {"size": 11, "weight": "bold"}
    TITLE_BAND = {"size": 22, "weight": "bold"}


class Spacing:
    """Pixel paddings reused across screen layouts."""

    SCREEN_PAD_X = 16
    SCREEN_PAD_Y = (14, 4)
    HEADER_TITLE_PAD = 12
    BODY_PAD_X = 32
    # Horizontal inset of the practice card from the window edges.
    CARD_PAD_X = 110
    CARD_RADIUS = 18


class WindowSizes:
    """Initial window geometries."""

    MAIN = "900x600"
    # Below this size the whole UI scales down with the window instead of
    # clipping, bottoming out at ``MIN_UI_SCALE`` (the hard window minimum).
    # The width leaves headroom over the widest header (~700 px): text renders
    # up to ~7% wider at small scales, since font sizes round to whole points.
    MAIN_MIN = (760, 450)
    MIN_UI_SCALE = 0.4
    # UI scale is rounded to this step so a drag doesn't redraw on every pixel.
    UI_SCALE_STEP = 0.05
    # Wait this long after the last resize event before rescaling.
    RESCALE_DEBOUNCE_MS = 80
    NEW_DB_DIALOG = "420x260"
    WORD_EDIT_DIALOG = "440x260"


class PollIntervals:
    """``after()`` delays for background-job pollers."""

    MS = 50


class Defaults:
    """Default values for GUI inputs."""

    EPOCHS = 100


class Limits:
    """Inclusive (min, max) ranges for validated inputs."""

    RECALL_THRESHOLD = (0.50, 0.95)
    RECALL_THRESHOLD_STEPS = 45
    MAX_INTERVAL_DAYS = (1, 3650)
    EPOCHS_MIN = 1


class Colors:
    """Color tokens for both light and dark mode.

    A deep burgundy carries the brand; green is kept for remembered / on track
    and a brighter red for forgot / overdue, so neither is ever decoration. The
    two are also kept apart in lightness (dark red, light green) so they stay
    distinguishable for red-green colour-blind users. ``(light, dark)`` tuples
    are customtkinter's per-mode colour form; widget defaults live in
    :data:`Paths.COLOR_THEME`.
    """

    # Surfaces
    GROUND = ("#F7F0F0", "#170809")
    SURFACE = ("#FFFFFF", "#2A0F13")
    BORDER = ("#EBD5D7", "#44191F")
    TEXT = ("#2B0D11", "#F4E6E7")
    TEXT_MUTED = ("#6B4A4E", "#BF9599")
    CHIP_BG = ("#F6E1E3", "#44191F")
    # Card outline: invisible on light, a faint edge on dark. Recoloured green
    # or red on the practice card to show an answer's outcome.
    CARD_BORDER = ("#FFFFFF", "#44191F")

    # Brand band (header strip, practice background)
    BRAND = "#4A1018"
    BRAND_HOVER = "#6E1E28"
    ON_BRAND = "#FFFFFF"
    ON_BRAND_MUTED = "#E3B5BA"
    ON_BRAND_BORDER = "#7A3A43"
    PRACTICE_BG = (BRAND, "#170809")
    PROGRESS_TRACK = ("#6A2730", "#3A1418")

    # Call to action on the brand band (the word list's Practice button)
    ACCENT = "#F4A3AB"
    ACCENT_HOVER = "#F8C1C7"
    ON_ACCENT = "#2B0D11"

    # Remembered / forgot
    GREEN = "#7ED69B"
    ON_GREEN = "#08170F"
    GREEN_TINT = ("#DDF2E3", "#1E4D33")
    GREEN_TINT_HOVER = ("#C8EBD3", "#285E40")
    GREEN_TINT_TEXT = ("#14482C", "#A6E6BA")
    RED = ("#B3261E", "#C62F26")
    RED_TINT = ("#FBE4E1", "#6B1E22")
    RED_TINT_HOVER = ("#F6CFCA", "#7E262B")
    RED_TINT_TEXT = ("#8C1D18", "#FFB4AB")
    # "Bad" red that still reads on the burgundy band in both modes.
    RED_ON_BRAND = "#FF8A80"

    # Action accents
    DANGER = "#B3261E"
    DANGER_HOVER = "#8C1D18"

    # Treeview palette
    TREE_DARK_BG = "#2A0F13"
    TREE_DARK_FG = "#F4E6E7"
    TREE_DARK_SEL_BG = "#5A1F27"
    TREE_DARK_HEADING_BG = "#3A1418"

    TREE_LIGHT_BG = "#FFFFFF"
    TREE_LIGHT_FG = "#2B0D11"
    TREE_LIGHT_SEL_BG = "#F0C9CD"
    TREE_LIGHT_HEADING_BG = "#F6E1E3"

    TREE_ROW_HEIGHT = 30

    # Matplotlib plot palette
    PLOT_DARK_BG = "#170809"
    PLOT_DARK_AX = "#2A0F13"
    PLOT_DARK_TEXT = "#F4E6E7"

    PLOT_LIGHT_BG = "#F7F0F0"
    PLOT_LIGHT_AX = "#FFFFFF"
    PLOT_LIGHT_TEXT = "#2B0D11"

    # Train green, val red; val is also dashed so the two stay apart for
    # red-green colour-blind users, as both are similarly light on dark.
    PLOT_DARK_TRAIN_LINE = "#7ED69B"
    PLOT_DARK_VAL_LINE = "#FF8A80"
    PLOT_LIGHT_TRAIN_LINE = "#1E7A47"
    PLOT_LIGHT_VAL_LINE = "#B3261E"


class Hints:
    """Hint-bar strings shown across the practice screen state machine."""

    PROMPT_DOWN = "↓  show translation"
    REMEMBERED_BUTTON = "Remembered  →"
    FORGOT_BUTTON = "←  Didn't remember"
    RESULT_BAR = "·   any arrow for the next word"
    OFFER_EXTRA_BAR = "→  keep practicing     ←  back to the word list"
    DONE_BAR = "press ← Back to return to the word list"


class Paths:
    """On-disk locations used by GUI screens.

    Thin re-export shim over :mod:`src.paths` so the GUI keeps a single
    import (``from .theme import Paths``) and the canonical layout lives
    project-wide in :mod:`src.paths`.
    """

    STORAGE_DIR = _paths.STORAGE_DIR
    DATABASES_DIR = _paths.DATABASES_DIR
    MODELS_DIR = _paths.MODELS_DIR
    model_path = staticmethod(_paths.model_path)
    # customtkinter colour theme giving every widget the burgundy defaults.
    COLOR_THEME = Path(__file__).with_name("red_green.json")
