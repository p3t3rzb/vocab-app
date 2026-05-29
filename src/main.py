"""Application entry point.

Launches the customtkinter GUI for the vocab-repetition app. Run with:

    uv run python -m src.main
"""

import customtkinter as ctk

from src.gui.app import App

ctk.set_appearance_mode("system")
ctk.set_default_color_theme("blue")

app = App()
app.mainloop()
