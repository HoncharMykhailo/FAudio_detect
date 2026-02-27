import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pygame
import librosa
import librosa.display
import numpy as np
import scipy.fftpack
import matplotlib
import threading
import time
from pathlib import Path
import os

from tensorflow.keras.models import load_model

matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization


# --- ДОПОМІЖНА ФУНКЦІЯ LFCC ---
def extract_lfcc(y, sr, n_mfcc=40, n_filters=128):
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=512)) ** 2
    fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    filter_freqs = np.linspace(0, sr / 2, n_filters + 2)

    lin_filters = np.zeros((n_filters, len(fft_freqs)))
    for i in range(n_filters):
        lower, center, upper = filter_freqs[i], filter_freqs[i + 1], filter_freqs[i + 2]
        for j, f in enumerate(fft_freqs):
            if lower <= f <= center:
                lin_filters[i, j] = (f - lower) / (center - lower)
            elif center < f <= upper:
                lin_filters[i, j] = (upper - f) / (upper - center)

    linear_spec = np.dot(lin_filters, S)
    log_linear_spec = librosa.power_to_db(linear_spec)
    return scipy.fftpack.dct(log_linear_spec, axis=0, type=2, norm='ortho')[:n_mfcc]


class AudioAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Аналізатор аудіо з Ансамблем ШІ (Mel + LFCC)")
        self.root.geometry("1400x900")

        pygame.mixer.init(frequency=44100)

        # -------- ЗАВАНТАЖЕННЯ ШІ МОДЕЛЕЙ --------
        self.mel_model_path = "deepfake_cnn.keras"
        self.lfcc_model_path = "deepfake_lfcc_cnn.keras"
        self.mel_model = None
        self.lfcc_model = None
        self.load_ai_models()

        # -------- ДАНІ --------
        self.audio_path = None
        self.audio_data = None
        self.sample_rate = None
        self.duration = 0
        self.segments = []
        self.selected_segment = None

        # -------- ПЛЕЄР --------
        self.is_playing = False
        self.start_time = 0.0
        self.paused_at = 0.0
        self.current_time = 0.0

        self.create_widgets()
        self.update_ui_loop()

    def load_ai_models(self):
        print("Завантаження ШІ моделей...")
        if os.path.exists(self.mel_model_path):
            try:
                self.mel_model = load_model(self.mel_model_path)
                print("✅ Mel-CNN завантажено.")
            except Exception as e:
                print(f"Помилка Mel-моделі: {e}")

        if os.path.exists(self.lfcc_model_path):
            try:
                self.lfcc_model = load_model(self.lfcc_model_path)
                print("✅ LFCC-CNN завантажено.")
            except Exception as e:
                print(f"Помилка LFCC-моделі: {e}")

    # =====================================================
    # UI SETUP
    # =====================================================
    def create_widgets(self):
        main = tk.Frame(self.root)
        main.pack(fill="both", expand=True)

        self.upload_frame = tk.Frame(main)
        self.upload_frame.pack(fill="x", pady=20)
        tk.Button(self.upload_frame, text="Вибрати файл", command=self.load_file, bg="#667eea", fg="white",
                  font=("Arial", 12)).pack()

        status_text = "🟢 ШІ активний (Mel + LFCC)" if self.mel_model and self.lfcc_model else "⚠️ Працюють не всі моделі!"
        tk.Label(self.upload_frame, text=status_text,
                 fg="green" if self.mel_model and self.lfcc_model else "orange").pack(pady=5)

        self.player_frame = tk.Frame(main)
        self.file_label = tk.Label(self.player_frame, font=("Arial", 12, "bold"))
        self.file_label.pack(pady=5)

        controls = tk.Frame(self.player_frame)
        controls.pack()
        self.play_btn = tk.Button(controls, text="▶ Play", command=self.toggle_play, width=10)
        self.play_btn.pack(side="left", padx=5)
        tk.Button(controls, text="⏹ Stop", command=self.stop_audio, width=10).pack(side="left", padx=5)

        tk.Button(controls, text="📂 Новий файл", command=self.reset_app, width=12).pack(side="left", padx=5)

        self.time_label = tk.Label(controls, text="00:00 / 00:00")
        self.time_label.pack(side="left", padx=15)

        self.progress_bar = ttk.Scale(self.player_frame, from_=0, to=1, orient="horizontal")
        self.progress_bar.pack(fill="x", padx=20, pady=10)
        self.progress_bar.bind("<ButtonPress-1>", lambda e: setattr(self, 'is_dragging', True))
        self.progress_bar.bind("<ButtonRelease-1>", self.on_slider_release)
        self.is_dragging = False

        self.content_frame = tk.Frame(main)
        self.content_frame.pack(fill="both", expand=True)

        left_panel = tk.Frame(self.content_frame, width=450)
        left_panel.pack(side="left", fill="y", padx=10, pady=10)
        left_panel.pack_propagate(False)

        cols = ("id", "time", "mel_score", "lfcc_score")
        self.tree = ttk.Treeview(left_panel, columns=cols, show="headings")
        self.tree.heading("id", text="#")
        self.tree.heading("time", text="Час")
        self.tree.heading("mel_score", text="Mel ШІ (%)")
        self.tree.heading("lfcc_score", text="LFCC ШІ (%)")

        self.tree.column("id", width=30)
        self.tree.column("time", width=80)
        self.tree.column("mel_score", width=80)
        self.tree.column("lfcc_score", width=80)

        self.tree.pack(side="left", fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.on_segment_select)

        self.plot_container = tk.Frame(self.content_frame)
        self.plot_container.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.canvas_scroll = tk.Canvas(self.plot_container, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.plot_container, orient="vertical", command=self.canvas_scroll.yview)
        self.scrollable_frame = tk.Frame(self.canvas_scroll)

        self.scrollable_frame.bind("<Configure>",
                                   lambda e: self.canvas_scroll.configure(scrollregion=self.canvas_scroll.bbox("all")))
        self.canvas_scroll.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas_scroll.configure(yscrollcommand=self.scrollbar.set)

        self.canvas_scroll.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.canvas_scroll.bind_all("<MouseWheel>", self._on_mousewheel)

        self.fig = Figure(figsize=(8, 16), dpi=100)
        self.ax_wave = self.fig.add_subplot(5, 1, 1)
        self.ax_spec = self.fig.add_subplot(5, 1, 2, sharex=self.ax_wave)
        self.ax_fft = self.fig.add_subplot(5, 1, 3)
        self.ax_mfcc = self.fig.add_subplot(5, 1, 4, sharex=self.ax_wave)
        self.ax_lfcc = self.fig.add_subplot(5, 1, 5, sharex=self.ax_wave)
        self.fig.tight_layout(pad=4.0)

        self.playheads = {}
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, master=self.scrollable_frame)
        self.mpl_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _on_mousewheel(self, event):
        self.canvas_scroll.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # =====================================================
    # LOGIC: PREDICTION & ANALYSIS
    # =====================================================
    def predict_ai_ensemble(self, chunk, original_sr):
        mel_pred = None
        lfcc_pred = None

        chunk_22k = librosa.resample(y=chunk, orig_sr=original_sr, target_sr=22050) if original_sr != 22050 else chunk

        target_len = 22050 * 1

        if len(chunk_22k) > target_len:
            chunk_22k = chunk_22k[:target_len]
        else:
            chunk_22k = np.pad(chunk_22k, (0, max(0, target_len - len(chunk_22k))), "constant")

        # 1.  Mel
        if self.mel_model:
            mel = librosa.feature.melspectrogram(y=chunk_22k, sr=22050, n_mels=128, fmax=8000)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_input = mel_db[np.newaxis, ..., np.newaxis]
            mel_pred = float(self.mel_model.predict(mel_input, verbose=0)[0][0])

        # 2. LFCC
        if self.lfcc_model:
            lfcc_feat = extract_lfcc(chunk_22k, 22050, n_mfcc=40, n_filters=128)
            lfcc_input = lfcc_feat[np.newaxis, ..., np.newaxis]
            lfcc_pred = float(self.lfcc_model.predict(lfcc_input, verbose=0)[0][0])

        return mel_pred, lfcc_pred

    def analyze_audio(self):
        y, sr = librosa.load(self.audio_path, sr=None)
        self.audio_data, self.sample_rate, self.duration = y, sr, len(y) / sr
        self.root.after(0, lambda: self.progress_bar.config(to=self.duration))

        step = int(1.0 * sr)
        self.segments.clear()

        for i in range(0, len(y), step):
            chunk = y[i:i + step]
            if len(chunk) < step // 2: continue

            rms = float(np.sqrt(np.mean(chunk ** 2)))
            mel_score, lfcc_score = (None, None)

            if rms >= 0.005:
                mel_score, lfcc_score = self.predict_ai_ensemble(chunk, sr)

            self.segments.append({
                "id": len(self.segments) + 1, "start": i / sr, "end": (i + step) / sr,
                "data": chunk, "sr": sr, "rms": rms,
                "mel_score": mel_score, "lfcc_score": lfcc_score
            })
        self.root.after(0, self.populate_table)

    def populate_table(self):
        for item in self.tree.get_children(): self.tree.delete(item)
        for seg in self.segments:
            mel_val = seg["mel_score"]
            lfcc_val = seg["lfcc_score"]

            mel_text = f"{mel_val * 100:.1f}%" if mel_val is not None else "---"
            lfcc_text = f"{lfcc_val * 100:.1f}%" if lfcc_val is not None else "---"

            tag = "normal"
            if mel_val is not None and lfcc_val is not None:
                if mel_val > 0.6 or lfcc_val > 0.6:
                    if abs(mel_val - lfcc_val) > 0.4:
                        tag = "suspicious"  # Моделі не згодні
                    else:
                        tag = "fake"  # Обидві впевнені, що фейк
                else:
                    tag = "real"  # Обидві впевнені, що справжнє

            self.tree.insert("", "end", values=(seg["id"], f"{seg['start']:.1f}-{seg['end']:.1f}", mel_text, lfcc_text),
                             tags=(tag,))

        self.tree.tag_configure("fake", background="#ffcccc")  # Червоний
        self.tree.tag_configure("real", background="#ccffcc")  # Зелений
        self.tree.tag_configure("suspicious", background="#fff2cc")  # Жовтий (Конфлікт моделей)


    def load_file(self):
        path = filedialog.askopenfilename()
        if not path: return
        self.audio_path = path
        self.file_label.config(text=f"Файл: {Path(path).name}")
        self.upload_frame.pack_forget()
        self.player_frame.pack(fill="x", pady=10)
        self.content_frame.pack(fill="both", expand=True)
        pygame.mixer.music.load(path)
        threading.Thread(target=self.analyze_audio, daemon=True).start()

    def on_segment_select(self, event):
        sel = self.tree.selection()
        if not sel: return
        seg_id = int(self.tree.item(sel[0])["values"][0])
        self.selected_segment = next((s for s in self.segments if s["id"] == seg_id), None)
        if self.selected_segment: self.plot_segment()

    def plot_segment(self):
        seg, data, sr = self.selected_segment, self.selected_segment["data"], self.selected_segment["sr"]
        duration = len(data) / sr

        for ax in [self.ax_wave, self.ax_spec, self.ax_fft, self.ax_mfcc, self.ax_lfcc]: ax.clear()

        # 1. Wave
        t = np.linspace(0, duration, len(data))
        self.ax_wave.plot(t, data, color="#4F46E5", lw=0.8)
        self.ax_wave.set_title("Форма хвилі", fontsize=10)
        self.ax_wave.set_xlim(0, duration)

        # 2. Spec
        D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="hz", ax=self.ax_spec, cmap='magma')
        self.ax_spec.set_title("Спектрограма", fontsize=10)

        # 3. FFT
        fft = np.fft.rfft(data)
        freq = np.fft.rfftfreq(len(data), 1 / sr)
        self.ax_fft.plot(freq, np.abs(fft), color="#059669")
        self.ax_fft.set_title("FFT Спектр", fontsize=10)
        self.ax_fft.set_xlim(0, 5000)

        # 4. MFCC
        mfcc_data = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13)
        librosa.display.specshow(mfcc_data, sr=sr, x_axis="time", ax=self.ax_mfcc, cmap='coolwarm')
        self.ax_mfcc.set_title("MFCC (Мел-кепстр)", fontsize=10)

        # 5. LFCC
        lfcc_data = extract_lfcc(data, sr, n_mfcc=13, n_filters=40)
        librosa.display.specshow(lfcc_data, sr=sr, x_axis="time", ax=self.ax_lfcc, cmap='viridis')
        self.ax_lfcc.set_title("LFCC", fontsize=10)

        self.playheads = {
            'wave': self.ax_wave.axvline(x=0, color='red', lw=1.5),
            'spec': self.ax_spec.axvline(x=0, color='red', lw=1.5),
            'mfcc': self.ax_mfcc.axvline(x=0, color='red', lw=1.5),
            'lfcc': self.ax_lfcc.axvline(x=0, color='red', lw=1.5)
        }
        self.fig.tight_layout(pad=3.0)
        self.mpl_canvas.draw()

    def toggle_play(self):
        if self.is_playing:
            pygame.mixer.music.pause()
            self.paused_at += time.time() - self.start_time
            self.is_playing = False
        else:
            if self.paused_at == 0:
                pygame.mixer.music.play()
            else:
                pygame.mixer.music.unpause()
            self.start_time, self.is_playing = time.time(), True

    def stop_audio(self):
        pygame.mixer.music.stop()
        self.is_playing, self.paused_at, self.current_time = False, 0, 0
        self.progress_bar.set(0)
        for ph in self.playheads.values(): ph.set_xdata([-1, -1])
        self.mpl_canvas.draw_idle()

    def reset_app(self):
        self.stop_audio()
        self.segments.clear()
        self.selected_segment = None
        self.audio_path = None

        for item in self.tree.get_children():
            self.tree.delete(item)

        for ax in [self.ax_wave, self.ax_spec, self.ax_fft, self.ax_mfcc, self.ax_lfcc]:
            ax.clear()
        self.mpl_canvas.draw()

        self.player_frame.pack_forget()
        self.content_frame.pack_forget()
        self.upload_frame.pack(fill="x", pady=20)

    def on_slider_release(self, event):
        self.is_dragging = False
        self.paused_at = self.progress_bar.get()
        if self.is_playing:
            pygame.mixer.music.play(start=self.paused_at)
            self.start_time = time.time()


    def update_ui_loop(self):
        if self.is_playing:
            self.current_time = self.paused_at + (time.time() - self.start_time)
            if self.current_time >= self.duration:
                self.stop_audio()
            else:
                if not self.is_dragging: self.progress_bar.set(self.current_time)
                self.time_label.config(
                    text=f"{int(self.current_time // 60)}:{int(self.current_time % 60):02d} / {int(self.duration // 60)}:{int(self.duration % 60):02d}")
                if self.selected_segment:
                    local_time = self.current_time - self.selected_segment["start"]
                    if 0 <= local_time <= (self.selected_segment["end"] - self.selected_segment["start"]):
                        for ph in self.playheads.values(): ph.set_xdata([local_time, local_time])
                        self.mpl_canvas.draw_idle()
        self.root.after(50, self.update_ui_loop)


if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style();
    style.theme_use('clam')
    app = AudioAnalyzerApp(root)
    root.mainloop()