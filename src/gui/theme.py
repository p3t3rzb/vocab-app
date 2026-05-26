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


class Fonts:
    """Keyword args for :class:`customtkinter.CTkFont`."""

    TITLE = {"size": 22, "weight": "bold"}
    HEADER = {"size": 18, "weight": "bold"}
    SECTION = {"size": 14, "weight": "bold"}
    BODY = {"size": 14}
    BODY_BOLD = {"size": 14, "weight": "bold"}
    SMALL = {"size": 13}
    PROMPT = {"size": 36, "weight": "bold"}
    ANSWER = {"size": 32}


class Spacing:
    """Pixel paddings reused across screen layouts."""

    SCREEN_PAD_X = 16
    SCREEN_PAD_Y = (14, 4)
    HEADER_TITLE_PAD = 12
    BODY_PAD_X = 32
    SECTION_GAP = 20
    FIELD_GAP = 6
    BUTTON_GAP = 6


class WindowSizes:
    """Initial window geometries."""

    MAIN = "900x600"
    MAIN_MIN = (700, 450)
    NEW_DB_DIALOG = "420x260"
    WORD_EDIT_DIALOG = "440x220"


class PollIntervals:
    """``after()`` delays for background-job pollers."""

    MS = 50


class Defaults:
    """Default values for GUI inputs and the app itself."""

    EPOCHS = 100
    APPEARANCE_MODE = "system"
    COLOR_THEME = "blue"


class Limits:
    """Inclusive (min, max) ranges for validated inputs."""

    RECALL_THRESHOLD = (0.50, 0.95)
    RECALL_THRESHOLD_STEPS = 45
    MAX_INTERVAL_DAYS = (1, 3650)
    EPOCHS_MIN = 1


class Colors:
    """Color tokens for both light and dark mode."""

    # Action accents
    DANGER = "#c0392b"
    DANGER_HOVER = "#922b21"

    # Treeview palette
    TREE_DARK_BG = "#2b2b2b"
    TREE_DARK_FG = "#dce4ee"
    TREE_DARK_SEL_BG = "#1f6aa5"
    TREE_DARK_HEADING_BG = "#1a1a2e"

    TREE_LIGHT_BG = "#f0f0f0"
    TREE_LIGHT_FG = "#1a1a1a"
    TREE_LIGHT_SEL_BG = "#1f6aa5"
    TREE_LIGHT_HEADING_BG = "#dde1e7"

    TREE_ROW_HEIGHT = 28

    # Matplotlib plot palette
    PLOT_DARK_BG = "#2b2b2b"
    PLOT_DARK_AX = "#3a3a3a"
    PLOT_DARK_TEXT = "#dce4ee"

    PLOT_LIGHT_BG = "#f5f5f5"
    PLOT_LIGHT_AX = "#ffffff"
    PLOT_LIGHT_TEXT = "#1a1a1a"

    PLOT_TRAIN_LINE = "steelblue"
    PLOT_VAL_LINE = "orange"


class Hints:
    """Hint-bar strings shown across the practice screen state machine."""

    PROMPT_DOWN = "↓  show translation"
    ANSWER_BAR = "←  didn't remember     →  remembered"
    SAVING_BAR = "Saving…"
    RESULT_BAR = "press any arrow for the next word"
    DONE_BAR = "press ← Back to return to the word list"


class Paths:
    """On-disk locations used by GUI screens."""

    STORAGE_DIR = Path(__file__).parent.parent.parent / "storage"
    MODELS_DIR = STORAGE_DIR / "models"

    @staticmethod
    def model_path(src_lang: str, tgt_lang: str) -> Path:
        return Paths.MODELS_DIR / f"{src_lang.lower()}_{tgt_lang.lower()}.pt"
