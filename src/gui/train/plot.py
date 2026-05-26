"""Live training-loss plot rendered via the Agg backend.

Owns the matplotlib :class:`Figure` and the two loss lines. The figure is
rendered to a PNG byte buffer and decoded by ``tk.PhotoImage`` — avoiding
the TkAgg C extension so the plot works with the pure-Python Tcl/Tk that
ships with uv.
"""
from __future__ import annotations

import base64
import io
import tkinter as tk

import customtkinter as ctk
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ..theme import Colors


class LossPlot:
    """Encapsulates the matplotlib figure used by the training screen."""

    def __init__(self, target: tk.Label) -> None:
        self._target = target
        self._epochs: list[int] = []
        self._train_losses: list[float] = []
        self._val_losses: list[float] = []
        self._pending_after_id: str | None = None
        self._destroyed = False
        self._build_figure()

    # ------------------------------------------------------------------
    # Figure setup
    # ------------------------------------------------------------------

    def _build_figure(self) -> None:
        """Create the figure and configure axes/colours for current appearance."""
        dark = ctk.get_appearance_mode() == "Dark"
        fig_bg = Colors.PLOT_DARK_BG if dark else Colors.PLOT_LIGHT_BG
        ax_bg = Colors.PLOT_DARK_AX if dark else Colors.PLOT_LIGHT_AX
        text_c = Colors.PLOT_DARK_TEXT if dark else Colors.PLOT_LIGHT_TEXT

        self._target.configure(bg=fig_bg)

        self._fig = Figure(dpi=100)
        FigureCanvasAgg(self._fig)
        self._fig.patch.set_facecolor(fig_bg)

        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor(ax_bg)
        self._ax.set_xlabel("Epoch", color=text_c)
        self._ax.set_ylabel("Loss", color=text_c)
        self._ax.set_title("Training Progress", color=text_c)
        self._ax.tick_params(colors=text_c)
        for spine in self._ax.spines.values():
            spine.set_edgecolor(text_c)
        (self._line_train,) = self._ax.plot(
            [], [], color=Colors.PLOT_TRAIN_LINE, label="Train loss", linewidth=2,
        )
        (self._line_val,) = self._ax.plot(
            [], [], color=Colors.PLOT_VAL_LINE, label="Val loss", linewidth=2,
        )
        self._ax.legend(facecolor=ax_bg, labelcolor=text_c)
        self._fig.tight_layout()

    # ------------------------------------------------------------------
    # Data updates
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all data and redraw an empty plot."""
        self._epochs.clear()
        self._train_losses.clear()
        self._val_losses.clear()
        self._line_train.set_data([], [])
        self._line_val.set_data([], [])
        self._ax.relim()
        self.render()

    def append(self, epoch: int, train_loss: float, val_loss: float) -> None:
        """Add one epoch's losses and re-render."""
        self._epochs.append(epoch)
        self._train_losses.append(train_loss)
        self._val_losses.append(val_loss)
        self._line_train.set_data(self._epochs, self._train_losses)
        self._line_val.set_data(self._epochs, self._val_losses)
        self._ax.relim()
        self._ax.autoscale_view()
        self.render()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> None:
        """Resize the figure to match the label and blit it as a PhotoImage.

        Re-tries via ``after`` if the underlying widget hasn't been laid
        out yet (its width/height stays under 10 px until Tk has measured
        it for the first time).
        """
        self._pending_after_id = None
        if self._destroyed:
            return

        w = self._target.winfo_width()
        h = self._target.winfo_height()
        if w < 10 or h < 10:
            self._pending_after_id = self._target.after(50, self.render)
            return

        self._fig.set_size_inches(w / self._fig.dpi, h / self._fig.dpi)
        self._fig.tight_layout()
        self._fig.canvas.draw()

        buf = io.BytesIO()
        self._fig.savefig(buf, format="png")
        buf.seek(0)
        png_b64 = base64.b64encode(buf.getvalue()).decode()

        photo = tk.PhotoImage(data=png_b64)
        self._target.configure(image=photo)
        self._target._photo = photo  # prevent GC

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def destroy(self) -> None:
        """Cancel pending render callbacks and free the matplotlib figure."""
        if self._destroyed:
            return
        self._destroyed = True
        if self._pending_after_id is not None:
            try:
                self._target.after_cancel(self._pending_after_id)
            except Exception:
                pass
            self._pending_after_id = None
        # Drop the PhotoImage reference held on the label so it can be GC'd.
        if hasattr(self._target, "_photo"):
            try:
                self._target.configure(image="")
            except Exception:
                pass
            try:
                del self._target._photo
            except Exception:
                pass
        plt.close(self._fig)
