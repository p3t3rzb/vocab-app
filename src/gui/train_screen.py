from __future__ import annotations

import base64
import io
import queue
import threading
from pathlib import Path
from tkinter import messagebox
import tkinter as tk
from typing import TYPE_CHECKING

import customtkinter as ctk
import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

if TYPE_CHECKING:
    from .app import App


class TrainScreen(ctk.CTkFrame):
    _POLL_MS = 100

    def __init__(
        self, master: App, db_path: Path, src_lang: str, tgt_lang: str
    ) -> None:
        super().__init__(master, corner_radius=0)
        self._app = master
        self._db_path = db_path
        self._src_lang = src_lang
        self._tgt_lang = tgt_lang

        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._schedule_stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._train_losses: list[float] = []
        self._val_losses: list[float] = []
        self._epoch_nums: list[int] = []
        self._training = False
        self._computing_schedules = False

        self._build_ui()
        self._build_figure()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 4))
        header.grid_columnconfigure(1, weight=1)

        ctk.CTkButton(header, text="← Back", width=80, command=self._go_back).grid(
            row=0, column=0, sticky="w"
        )

        ctk.CTkLabel(
            header,
            text=f"Train Model  —  {self._src_lang} ↔ {self._tgt_lang}",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).grid(row=0, column=1, padx=12, sticky="w")

        # Plot canvas (native tk.Label with PhotoImage — no PIL ImageTk needed)
        plot_frame = ctk.CTkFrame(self, fg_color="transparent")
        plot_frame.grid(row=1, column=0, sticky="nsew", padx=16, pady=(4, 4))
        plot_frame.grid_rowconfigure(0, weight=1)
        plot_frame.grid_columnconfigure(0, weight=1)

        self._plot_label = tk.Label(plot_frame, borderwidth=0, bg="#2b2b2b")
        self._plot_label.grid(row=0, column=0, sticky="nsew")

        # Status
        self._status_var = ctk.StringVar(
            value="Ready. Set epoch count and press Train."
        )
        ctk.CTkLabel(
            self, textvariable=self._status_var, font=ctk.CTkFont(size=13)
        ).grid(row=2, column=0, sticky="ew", padx=16, pady=(2, 2))

        # Progress bar
        self._progress = ctk.CTkProgressBar(self, mode="indeterminate")
        self._progress.grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 4))
        self._progress.set(0)

        # Controls
        ctrl = ctk.CTkFrame(self, fg_color="transparent")
        ctrl.grid(row=4, column=0, pady=(4, 16))

        ctk.CTkLabel(ctrl, text="Epochs:").pack(side="left", padx=(0, 4))
        self._epochs_var = ctk.StringVar(value="100")
        self._epochs_entry = ctk.CTkEntry(ctrl, textvariable=self._epochs_var, width=60)
        self._epochs_entry.pack(side="left", padx=(0, 20))

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

    def _build_figure(self) -> None:
        dark = ctk.get_appearance_mode() == "Dark"
        fig_bg = "#2b2b2b" if dark else "#f5f5f5"
        ax_bg = "#3a3a3a" if dark else "#ffffff"
        text_c = "#dce4ee" if dark else "#1a1a1a"

        self._plot_label.configure(bg=fig_bg)

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
            [], [], color="steelblue", label="Train loss", linewidth=2
        )
        (self._line_val,) = self._ax.plot(
            [], [], color="orange", label="Val loss", linewidth=2
        )
        self._ax.legend(facecolor=ax_bg, labelcolor=text_c)
        self._fig.tight_layout()

        # Render an initial blank plot once the widget has a size
        self.after(100, self._render_plot)

    # ------------------------------------------------------------------
    # Plot rendering  (Agg → PNG bytes → base64 → tk.PhotoImage)
    # No PIL ImageTk, no TkAgg C extension — pure Tk 8.6 PNG support
    # ------------------------------------------------------------------

    def _render_plot(self) -> None:
        w = self._plot_label.winfo_width()
        h = self._plot_label.winfo_height()
        if w < 10 or h < 10:
            self.after(50, self._render_plot)
            return

        self._fig.set_size_inches(w / self._fig.dpi, h / self._fig.dpi)
        self._fig.tight_layout()
        self._fig.canvas.draw()

        buf = io.BytesIO()
        self._fig.savefig(buf, format="png")
        buf.seek(0)
        png_b64 = base64.b64encode(buf.getvalue()).decode()

        photo = tk.PhotoImage(data=png_b64)
        self._plot_label.configure(image=photo)
        self._plot_label._photo = photo  # prevent GC

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

                elif tag == "schedule_progress":
                    _, done, total = item
                    self._status_var.set(f"Computing schedules… {done}/{total}")

                elif tag == "done":
                    _, path = item
                    self._on_training_done(path)

                elif tag == "cancelled":
                    self._on_training_finished("Training cancelled.")
                    return

                elif tag == "error":
                    _, msg = item
                    self._on_training_finished(f"Error: {msg}", is_error=True)
                    return

                elif tag == "schedules_done":
                    self._on_schedules_done()
                    return

                elif tag == "schedules_cancelled":
                    self._on_training_finished("Schedule computation cancelled.")
                    return

                elif tag == "schedules_error":
                    _, msg = item
                    self._on_schedules_error(msg)
                    return

        except queue.Empty:
            pass

        if self._training or self._computing_schedules:
            self.after(self._POLL_MS, self._poll)

    def _on_training_done(self, checkpoint_path: str) -> None:
        self._training = False
        self._computing_schedules = True
        self._schedule_stop_event.clear()
        self._status_var.set("Computing schedules…")
        self._btn_cancel.configure(state="normal")

        from pathlib import Path as _Path

        model_path = _Path(checkpoint_path)

        self._thread = threading.Thread(
            target=self._schedule_worker,
            args=(model_path,),
            daemon=True,
        )
        self._thread.start()

    def _schedule_worker(self, model_path) -> None:
        try:
            from src.model.schedule import compute_all_schedules
            from src.settings import load_settings

            def on_progress(done: int, total: int) -> None:
                self._queue.put(("schedule_progress", done, total))

            compute_all_schedules(
                model_path=model_path,
                on_progress=on_progress,
                stop_event=self._schedule_stop_event,
                cfg=load_settings().to_predict_config(),
            )
            if self._schedule_stop_event.is_set():
                self._queue.put(("schedules_cancelled",))
            else:
                self._queue.put(("schedules_done",))
        except Exception as exc:
            self._queue.put(("schedules_error", str(exc)))

    def _on_schedules_done(self) -> None:
        self._computing_schedules = False
        self._progress.stop()
        self._progress.set(0)
        self._btn_train.configure(state="normal")
        self._epochs_entry.configure(state="normal")
        self._btn_cancel.configure(state="disabled")
        self._status_var.set("Done — model trained and schedules updated.")

    def _on_schedules_error(self, msg: str) -> None:
        self._computing_schedules = False
        self._progress.stop()
        self._progress.set(0)
        self._btn_train.configure(state="normal")
        self._epochs_entry.configure(state="normal")
        self._btn_cancel.configure(state="disabled")
        self._status_var.set(f"Schedule error: {msg}")
        messagebox.showerror("Schedule error", msg, parent=self)

    def _on_training_finished(self, message: str, is_error: bool = False) -> None:
        self._training = False
        self._computing_schedules = False
        self._progress.stop()
        self._progress.set(0)
        self._btn_train.configure(state="normal")
        self._epochs_entry.configure(state="normal")
        self._btn_cancel.configure(state="disabled")
        self._status_var.set(message)
        if is_error:
            messagebox.showerror("Training error", message, parent=self)

    def _cancel_training(self) -> None:
        if self._training:
            self._stop_event.set()
            self._btn_cancel.configure(state="disabled")
            self._status_var.set("Cancelling… waiting for current epoch to finish.")
        elif self._computing_schedules:
            self._schedule_stop_event.set()
            self._btn_cancel.configure(state="disabled")
            self._status_var.set("Cancelling schedule computation…")

    def _go_back(self) -> None:
        if self._training:
            confirmed = messagebox.askyesno(
                "Training in progress",
                "Training is still running. Cancel it and go back?",
                parent=self,
            )
            if not confirmed:
                return
            self._stop_event.set()
        elif self._computing_schedules:
            confirmed = messagebox.askyesno(
                "Computing schedules",
                "Schedule computation is running. Cancel it and go back?\n\n"
                "Due times will be updated next time you train.",
                parent=self,
            )
            if not confirmed:
                return
            self._schedule_stop_event.set()
            self._computing_schedules = False
        self._app.back_to_word_list()
