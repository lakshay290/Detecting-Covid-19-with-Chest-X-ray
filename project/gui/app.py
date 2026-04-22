import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from model.load_model import get_model
from model.predict import predict_image


class CovidDetectionApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("COVID-19 Detection using Chest X-ray")
        self.root.geometry("1040x700")
        self.root.minsize(920, 620)
        self.root.configure(bg="#e8edf5")

        self.colors = {
            "bg": "#e8edf5",
            "card": "#f9fbff",
            "border": "#d9e2ef",
            "text": "#102a43",
            "muted": "#627d98",
            "accent": "#0b7285",
            "accent_hover": "#095c6b",
            "cta": "#1d4ed8",
            "cta_hover": "#1e40af",
            "neutral": "#8a9aab",
            "neutral_hover": "#6b7c90",
            "preview_bg": "#dce5f2",
            "result_bg": "#eef4fb",
            "positive": "#b91c1c",
            "normal": "#15803d",
        }

        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure(
            "Covid.Horizontal.TProgressbar",
            troughcolor="#d6deea",
            background="#0b7285",
            bordercolor="#d6deea",
            lightcolor="#0b7285",
            darkcolor="#0b7285",
            thickness=8,
        )

        self.model_path = Path(__file__).resolve().parents[1] / "covid_model.h5"
        self.model = get_model(self.model_path)

        self.image_path: str | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None

        self.result_var = tk.StringVar(value="Result: --")
        self.confidence_var = tk.StringVar(value="Confidence: --")
        self.status_var = tk.StringVar(value="Ready")

        self._build_layout()

    def _build_layout(self) -> None:
        container = tk.Frame(self.root, bg=self.colors["bg"])
        container.pack(fill="both", expand=True, padx=20, pady=20)

        header_card = tk.Frame(
            container,
            bg=self.colors["card"],
            highlightthickness=1,
            highlightbackground=self.colors["border"],
        )
        header_card.pack(fill="x", pady=(0, 14))

        badge = tk.Label(
            header_card,
            text="AI DIAGNOSTIC ASSISTANT",
            font=("Segoe UI", 9, "bold"),
            bg=self.colors["card"],
            fg=self.colors["accent"],
            padx=4,
            pady=8,
        )
        badge.pack(anchor="w", padx=16)

        header = tk.Label(
            header_card,
            text="COVID-19 Detection using Chest X-ray",
            font=("Segoe UI", 24, "bold"),
            bg=self.colors["card"],
            fg=self.colors["text"],
        )
        header.pack(anchor="w", padx=16, pady=(0, 2))

        subheader = tk.Label(
            header_card,
            text="Upload a chest X-ray image and run an instant COVID-19 screening prediction.",
            font=("Segoe UI", 10),
            bg=self.colors["card"],
            fg=self.colors["muted"],
        )
        subheader.pack(anchor="w", padx=16, pady=(0, 12))

        body = tk.Frame(container, bg=self.colors["bg"])
        body.pack(fill="both", expand=True)

        left_panel = tk.Frame(
            body,
            bg=self.colors["card"],
            bd=0,
            relief="flat",
            highlightthickness=1,
            highlightbackground=self.colors["border"],
        )
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))

        right_panel = tk.Frame(
            body,
            bg=self.colors["card"],
            bd=0,
            relief="flat",
            highlightthickness=1,
            highlightbackground=self.colors["border"],
        )
        right_panel.pack(side="right", fill="both", expand=True, padx=(10, 0))

        preview_title = tk.Label(
            left_panel,
            text="X-ray Preview",
            font=("Segoe UI", 12, "bold"),
            bg=self.colors["card"],
            fg=self.colors["text"],
        )
        preview_title.pack(anchor="w", padx=16, pady=(14, 6))

        self.preview_label = tk.Label(
            left_panel,
            text="No image selected",
            bg=self.colors["preview_bg"],
            fg=self.colors["muted"],
            font=("Segoe UI", 12),
            width=42,
            height=20,
            anchor="center",
            highlightthickness=1,
            highlightbackground=self.colors["border"],
        )
        self.preview_label.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        action_title = tk.Label(
            right_panel,
            text="Actions",
            font=("Segoe UI", 12, "bold"),
            bg=self.colors["card"],
            fg=self.colors["text"],
        )
        action_title.pack(anchor="w", padx=16, pady=(14, 8))

        action_frame = tk.Frame(right_panel, bg=self.colors["card"])
        action_frame.pack(fill="x", padx=16, pady=(16, 10))

        self.upload_btn = self._create_action_button(
            action_frame, "Upload X-ray Image", self.upload_image, self.colors["cta"], self.colors["cta_hover"]
        )
        self.upload_btn.pack(fill="x", pady=(0, 10))

        self.predict_btn = self._create_action_button(
            action_frame, "Predict", self.predict, self.colors["accent"], self.colors["accent_hover"]
        )
        self.predict_btn.pack(fill="x", pady=(0, 10))

        self.reset_btn = self._create_action_button(
            action_frame, "Reset", self.reset, self.colors["neutral"], self.colors["neutral_hover"]
        )
        self.reset_btn.pack(fill="x")

        self.loading_bar = ttk.Progressbar(right_panel, mode="indeterminate", style="Covid.Horizontal.TProgressbar")
        self.loading_bar.pack(fill="x", padx=16, pady=(8, 8))
        self.loading_bar.pack_forget()

        status_label = tk.Label(
            right_panel,
            textvariable=self.status_var,
            font=("Segoe UI", 10),
            bg=self.colors["card"],
            fg=self.colors["muted"],
            anchor="w",
        )
        status_label.pack(fill="x", padx=16, pady=(0, 12))

        result_card = tk.Frame(
            right_panel,
            bg=self.colors["result_bg"],
            highlightthickness=1,
            highlightbackground=self.colors["border"],
        )
        result_card.pack(fill="x", padx=16, pady=(0, 16))

        result_heading = tk.Label(
            result_card,
            text="Prediction Output",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors["result_bg"],
            fg=self.colors["muted"],
            pady=8,
        )
        result_heading.pack(fill="x")

        self.result_label = tk.Label(
            result_card,
            textvariable=self.result_var,
            font=("Segoe UI", 20, "bold"),
            bg=self.colors["result_bg"],
            fg=self.colors["text"],
            pady=10,
        )
        self.result_label.pack(fill="x")

        confidence_label = tk.Label(
            result_card,
            textvariable=self.confidence_var,
            font=("Segoe UI", 12),
            bg=self.colors["result_bg"],
            fg=self.colors["text"],
            pady=0,
        )
        confidence_label.pack(fill="x", pady=(0, 12))

        if self.model is None:
            self.status_var.set("Model not loaded: place a valid covid_model.h5 in the project root")

    def _create_action_button(
        self,
        parent: tk.Widget,
        text: str,
        command,
        bg: str,
        hover_bg: str,
    ) -> tk.Button:
        button = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg="white",
            activebackground=hover_bg,
            activeforeground="white",
            font=("Segoe UI", 12, "bold"),
            bd=0,
            padx=12,
            pady=11,
            cursor="hand2",
            relief="flat",
        )
        button.bind("<Enter>", lambda _event: button.configure(bg=hover_bg))
        button.bind("<Leave>", lambda _event: button.configure(bg=bg))
        return button

    def upload_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select Chest X-ray Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")],
        )

        if not file_path:
            return

        try:
            image = Image.open(file_path).convert("RGB")
            image.thumbnail((420, 420))
            self.preview_photo = ImageTk.PhotoImage(image)

            self.preview_label.configure(image=self.preview_photo, text="", bg="#ffffff")
            self.image_path = file_path
            self.result_var.set("Result: --")
            self.confidence_var.set("Confidence: --")
            self.status_var.set("Image loaded")
            self.result_label.configure(fg=self.colors["text"])
        except Exception:
            self.image_path = None
            self.preview_photo = None
            messagebox.showerror("Invalid File", "Please select a valid image file.")
            self.status_var.set("Failed to load image")

    def predict(self) -> None:
        if not self.image_path:
            messagebox.showwarning("No Image", "Please upload an X-ray image before prediction.")
            return

        self._set_controls_state("disabled")
        self.loading_bar.pack(fill="x", padx=16, pady=(8, 8))
        self.loading_bar.start(10)
        self.status_var.set("Predicting...")

        thread = threading.Thread(target=self._predict_worker, daemon=True)
        thread.start()

    def _predict_worker(self) -> None:
        try:
            label, confidence = predict_image(self.model, self.image_path)
            self.root.after(0, lambda: self._on_prediction_success(label, confidence))
        except Exception as exc:
            self.root.after(0, lambda: self._on_prediction_error(str(exc)))

    def _on_prediction_success(self, label: str, confidence: float) -> None:
        self.loading_bar.stop()
        self.loading_bar.pack_forget()
        self._set_controls_state("normal")

        self.result_var.set(f"Result: {label}")
        self.confidence_var.set(f"Confidence: {confidence * 100:.2f}%")

        if label == "COVID Positive":
            self.result_label.configure(fg=self.colors["positive"])
        else:
            self.result_label.configure(fg=self.colors["normal"])

        self.status_var.set("Prediction complete")

    def _on_prediction_error(self, error_message: str) -> None:
        self.loading_bar.stop()
        self.loading_bar.pack_forget()
        self._set_controls_state("normal")
        self.status_var.set("Prediction failed")
        messagebox.showerror("Prediction Error", error_message)

    def _set_controls_state(self, state: str) -> None:
        self.upload_btn.configure(state=state)
        self.predict_btn.configure(state=state)
        self.reset_btn.configure(state=state)

    def reset(self) -> None:
        self.image_path = None
        self.preview_photo = None
        self.preview_label.configure(
            image="",
            text="No image selected",
            bg=self.colors["preview_bg"],
            fg=self.colors["muted"],
        )
        self.result_var.set("Result: --")
        self.confidence_var.set("Confidence: --")
        self.status_var.set("Ready")
        self.result_label.configure(fg=self.colors["text"])

    def run(self) -> None:
        self.root.mainloop()
