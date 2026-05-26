from __future__ import annotations

import io
import queue
import threading
from pathlib import Path
from tkinter import messagebox

import customtkinter as ctk
import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image as _PILImage


class TrainModelDialog(ctk.CTkToplevel):
    _POLL_MS = 100
    _PLOT_W = 680
    _PLOT_H = 360

    def __init__(self, master: ctk.CTkFrame, db_path: Path) -> None:
        super().__init__(master)
        self._db_path = db_path
        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._train_losses: list[float] = []
        self._val_losses: list[float] = []
        self._epoch_nums: list[int] = []
        self._training = False

        self.title("Train Model")
        self.geometry(f"{self._PLOT_W + 40}x{self._PLOT_H + 160}")
        self.resizable(True, True)

        self.transient(master)
        self.lift()
        self.focus_force()

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        dark = ctk.get_appearance_mode() == "Dark"
        self._fig_bg = "#2b2b2b" if dark else "#f5f5f5"
        self._ax_bg = "#3a3a3a" if dark else "#ffffff"
        self._text_c = "#dce4ee" if dark else "#1a1a1a"

        # --- matplotlib figure (Agg backend — no TkAgg C extension needed) ---
        self._fig = Figure(figsize=(self._PLOT_W / 100, self._PLOT_H / 100), dpi=100)
        FigureCanvasAgg(self._fig)  # attach Agg renderer
        self._fig.patch.set_facecolor(self._fig_bg)

        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor(self._ax_bg)
        self._ax.set_xlabel("Epoch", color=self._text_c)
        self._ax.set_ylabel("Loss", color=self._text_c)
        self._ax.set_title("Training Progress", color=self._text_c)
        self._ax.tick_params(colors=self._text_c)
        for spine in self._ax.spines.values():
            spine.set_edgecolor(self._text_c)
        (self._line_train,) = self._ax.plot(
            [], [], color="steelblue", label="Train loss", linewidth=2
        )
        (self._line_val,) = self._ax.plot(
            [], [], color="orange", label="Val loss", linewidth=2
        )
        self._ax.legend(facecolor=self._ax_bg, labelcolor=self._text_c)
        self._fig.tight_layout()

        # Plot is displayed as a CTkImage inside a CTkLabel
        self._plot_label = ctk.CTkLabel(self, text="")
        self._plot_label.grid(row=0, column=0, sticky="nsew", padx=12, pady=(12, 4))
        self._render_plot()

        # --- status label ---
        self._status_var = ctk.StringVar(
            value="Ready. Configure epochs and press Train."
        )
        ctk.CTkLabel(
            self, textvariable=self._status_var, font=ctk.CTkFont(size=13)
        ).grid(row=1, column=0, sticky="ew", padx=16, pady=(4, 2))

        # --- progress bar ---
        self._progress = ctk.CTkProgressBar(self, mode="indeterminate")
        self._progress.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 4))
        self._progress.set(0)

        # --- controls row ---
        ctrl = ctk.CTkFrame(self, fg_color="transparent")
        ctrl.grid(row=3, column=0, pady=(4, 14))

        ctk.CTkLabel(ctrl, text="Epochs:").pack(side="left", padx=(0, 4))
        self._epochs_var = ctk.StringVar(value="100")
        self._epochs_entry = ctk.CTkEntry(ctrl, textvariable=self._epochs_var, width=60)
        self._epochs_entry.pack(side="left", padx=(0, 16))

        self._btn_train = ctk.CTkButton(
            ctrl, text="Train", width=100, command=self._start_training
        )
        self._btn_train.pack(side="left", padx=6)

        self._btn_cancel = ctk.CTkButton(
            ctrl,
            text="Cancel",
            width=100,
            state="disabled",
            fg_color="#c0392b",
            hover_color="#922b21",
            command=self._cancel_training,
        )
        self._btn_cancel.pack(side="left", padx=6)

        ctk.CTkButton(ctrl, text="Close", width=100, command=self._on_close).pack(
            side="left", padx=6
        )

    # ------------------------------------------------------------------
    # Plot rendering (Agg → PIL → CTkImage — no TkAgg extension needed)
    # ------------------------------------------------------------------

    def _render_plot(self) -> None:
        self._fig.canvas.draw()
        buf = io.BytesIO()
        self._fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        pil_img = _PILImage.open(buf).copy()
        w = max(self._plot_label.winfo_width(), self._PLOT_W)
        h = max(self._plot_label.winfo_height(), self._PLOT_H)
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(w, h))
        self._plot_label.configure(image=ctk_img)
        self._plot_label._ctk_image = ctk_img  # prevent GC

    def _update_plot(self) -> None:
        self._line_train.set_data(self._epoch_nums, self._train_losses)
        self._line_val.set_data(self._epoch_nums, self._val_losses)
        self._ax.relim()
        self._ax.autoscale_view()
        self._render_plot()

    def _reset_plot(self) -> None:
        self._line_train.set_data([], [])
        self._line_val.set_data([], [])
        self._ax.relim()
        self._render_plot()

    # ------------------------------------------------------------------
    # Training control
    # ------------------------------------------------------------------

    def _start_training(self) -> None:
        if self._training:
            return

        try:
            epochs = int(self._epochs_var.get())
            if epochs < 1:
                raise ValueError
        except ValueError:
            messagebox.showwarning(
                "Invalid input", "Epochs must be a positive integer.", parent=self
            )
            return

        self._training = True
        self._stop_event.clear()
        self._train_losses.clear()
        self._val_losses.clear()
        self._epoch_nums.clear()
        self._reset_plot()

        self._btn_train.configure(state="disabled")
        self._epochs_entry.configure(state="disabled")
        self._btn_cancel.configure(state="normal")
        self._status_var.set("Starting training…")
        self._progress.start()

        from src.model.config import TrainConfig

        cfg = TrainConfig(epochs=epochs)
        db_url = f"sqlite:///{self._db_path}"

        self._thread = threading.Thread(
            target=self._training_worker,
            args=(db_url, cfg),
            daemon=True,
        )
        self._thread.start()
        self._poll()

    def _training_worker(self, db_url: str, cfg) -> None:
        try:
            from src.model.train import train

            result_path = train(
                db_url=db_url,
                config=cfg,
                on_epoch=lambda e, tr, vl: self._queue.put(("epoch", e, tr, vl)),
                stop_event=self._stop_event,
            )
            if self._stop_event.is_set():
                self._queue.put(("cancelled", None))
            else:
                self._queue.put(("done", str(result_path)))
        except Exception as exc:
            self._queue.put(("error", str(exc)))

    def _poll(self) -> None:
        try:
            while True:
                item = self._queue.get_nowait()
                tag = item[0]

                if tag == "epoch":
                    _, epoch, tr_loss, vl_loss = item
                    self._epoch_nums.append(epoch)
                    self._train_losses.append(tr_loss)
                    self._val_losses.append(vl_loss)
                    self._update_plot()
                    total = self._epochs_var.get()
                    self._status_var.set(
                        f"Epoch {epoch}/{total} — Train: {tr_loss:.5f}  |  Val: {vl_loss:.5f}"
                    )

                elif tag == "done":
                    _, path = item
                    self._on_training_finished(f"Done. Checkpoint → {path}")
                    return

                elif tag == "cancelled":
                    self._on_training_finished("Training cancelled.")
                    return

                elif tag == "error":
                    _, msg = item
                    self._on_training_finished(f"Error: {msg}", is_error=True)
                    return

        except queue.Empty:
            pass

        if self._training:
            self.after(self._POLL_MS, self._poll)

    def _on_training_finished(self, message: str, is_error: bool = False) -> None:
        self._training = False
        self._progress.stop()
        self._progress.set(0)
        self._btn_train.configure(state="normal")
        self._epochs_entry.configure(state="normal")
        self._btn_cancel.configure(state="disabled")
        self._status_var.set(message)
        if is_error:
            messagebox.showerror("Training error", message, parent=self)

    def _cancel_training(self) -> None:
        if not self._training:
            return
        self._stop_event.set()
        self._btn_cancel.configure(state="disabled")
        self._status_var.set("Cancelling… waiting for current epoch to finish.")

    def _on_close(self) -> None:
        if self._training:
            confirmed = messagebox.askyesno(
                "Training in progress",
                "Training is still running. Cancel it and close?",
                parent=self,
            )
            if not confirmed:
                return
            self._stop_event.set()
        self.destroy()
