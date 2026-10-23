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

    The red-green palette: a deep forest green carries the brand, light green
    means remembered / on track, and red means forgot / overdue. The two are
    also kept far apart in lightness (dark red, light green) so they stay
    distinguishable for red-green colour-blind users. ``(light, dark)`` tuples
    are customtkinter's per-mode colour form; widget defaults live in
    :data:`Paths.COLOR_THEME`.
    """

    # Surfaces
    GROUND = ("#EEF4F0", "#08170F")
    SURFACE = ("#FFFFFF", "#10291C")
    BORDER = ("#D2E3D8", "#1C3D2A")
    TEXT = ("#0F3324", "#E3F2E8")
    TEXT_MUTED = ("#4D6355", "#93B5A0")
    CHIP_BG = ("#DDF2E3", "#1C3D2A")
    # Card outline: invisible on light, a faint edge on dark. Recoloured green
    # or red on the practice card to show an answer's outcome.
    CARD_BORDER = ("#FFFFFF", "#1C3D2A")

    # Brand band (header strip, practice background)
    BRAND = "#0F3324"
    BRAND_HOVER = "#1E5A3C"
    ON_BRAND = "#FFFFFF"
    ON_BRAND_MUTED = "#A3C7AF"
    ON_BRAND_BORDER = "#3F5E4C"
    PRACTICE_BG = (BRAND, "#08170F")
    PROGRESS_TRACK = ("#28493A", "#163624")

    # Remembered / forgot
    GREEN = "#7ED69B"
    GREEN_HOVER = "#A6E6BA"
    ON_GREEN = "#08170F"
    GREEN_TINT = ("#DDF2E3", "#1E4D33")
    GREEN_TINT_HOVER = ("#C8EBD3", "#285E40")
    GREEN_TINT_TEXT = ("#14482C", "#A6E6BA")
    RED = ("#B3261E", "#C62F26")
    RED_TINT = ("#FBE4E1", "#3A1714")
    RED_TINT_HOVER = ("#F6CFCA", "#4A1D19")
    RED_TINT_TEXT = ("#8C1D18", "#FF8A80")
    # Red that reads on the dark brand band in both modes.
    RED_ON_BRAND = "#FF8A80"

    # Action accents
    DANGER = "#B3261E"
    DANGER_HOVER = "#8C1D18"

    # Treeview palette
    TREE_DARK_BG = "#10291C"
    TREE_DARK_FG = "#E3F2E8"
    TREE_DARK_SEL_BG = "#285E40"
    TREE_DARK_HEADING_BG = "#163624"

    TREE_LIGHT_BG = "#FFFFFF"
    TREE_LIGHT_FG = "#0F3324"
    TREE_LIGHT_SEL_BG = "#C8EBD3"
    TREE_LIGHT_HEADING_BG = "#DDF2E3"

    TREE_ROW_HEIGHT = 30

    # Matplotlib plot palette
    PLOT_DARK_BG = "#08170F"
    PLOT_DARK_AX = "#10291C"
    PLOT_DARK_TEXT = "#E3F2E8"

    PLOT_LIGHT_BG = "#EEF4F0"
    PLOT_LIGHT_AX = "#FFFFFF"
    PLOT_LIGHT_TEXT = "#0F3324"

    PLOT_TRAIN_LINE = "steelblue"
    PLOT_VAL_LINE = "orange"


class Hints:
    """Hint-bar strings shown across the practice screen state machine."""

    PROMPT_DOWN = "↓  show translation"
    REMEMBERED_BUTTON = "Remembered  →"
    FORGOT_BUTTON = "←  Didn't remember"
    RESULT_BAR = "any arrow  →  next word"
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
    # customtkinter colour theme giving every widget the red-green defaults.
    COLOR_THEME = Path(__file__).with_name("red_green.json")
