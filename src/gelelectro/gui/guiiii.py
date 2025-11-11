import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from skimage import io
import numpy as np
from utils import GelPreprocessor


def main():
    root = tk.Tk()
    root.title("Gel Analysis Tool")

    # okno na celu obrazovku
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.geometry(f"{screen_width}x{screen_height}")

    preprocessor = GelPreprocessor()

    # uloženie obrázkov
    original_img = None
    processed_img = None

    # ====== FUNKCIE ======

    # nahratie obrázka
    def upload_image():
        nonlocal original_img
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.png *.tif *.tiff")]
        )
        if not file_path:
            return
        original_img = io.imread(file_path, as_gray=True)
        show_image(original_img, original_label)
        messagebox.showinfo("Obrázok", "Obrázok bol úspešne nahratý!")

    # spracovanie obrázka
    def preprocess_image():
        nonlocal processed_img
        if original_img is None:
            messagebox.showwarning("Upozornenie", "Najprv nahraj obrázok!")
            return
        processed_img = preprocessor.process_image(original_img)
        show_image(processed_img, processed_label)
        messagebox.showinfo("Hotovo", "Obrázok bol spracovaný!")

    # placeholder - analýza veľkosti
    def analyze_size():
        if processed_img is None:
            messagebox.showwarning("Upozornenie", "Najprv spracuj obrázok!")
            return
        messagebox.showinfo("Analýza veľkosti",
                            "Táto funkcia zatiaľ nie je implementovaná.")

    # placeholder - analýza koncentrácie
    def compute_concentration():
        if processed_img is None:
            messagebox.showwarning("Upozornenie", "Najprv spracuj obrázok!")
            return
        messagebox.showinfo("Analýza koncentrácie",
                            "Táto funkcia zatiaľ nie je implementovaná.")

    # placeholder - export reportu
    def export_report():
        messagebox.showinfo("Export",
                            "Export reportu zatiaľ nie je implementovaný.")

    # zobrazenie obrázka s automatickým zachovaním pomeru strán
    def show_image(img, label_widget, max_width=600, max_height=600):
        # Ak je float, previesť na uint8 len pre zobrazenie
        if np.issubdtype(img.dtype, np.floating):
            img_display = (img * 255).astype(np.uint8)
        else:
            img_display = img.copy()

        # Vytvorenie PIL Image z numpy array
        img_pil = Image.fromarray(img_display)

        # Zachovanie pomeru strán
        original_width, original_height = img_pil.size
        ratio = min(max_width / original_width, max_height / original_height)
        new_size = (int(original_width * ratio), int(original_height * ratio))
        img_pil = img_pil.resize(new_size, Image.Resampling.LANCZOS)

        # Zobrazenie v Tkinter
        img_tk = ImageTk.PhotoImage(img_pil)
        label_widget.config(image=img_tk)
        label_widget.image = img_tk

    # layout GUI
    title = tk.Label(root,
                     text="Analýza elektroforetických gélov",
                     font=("Arial", 20, "bold"))
    title.pack(pady=30)

    # rám s tlačidlami
    button_frame = tk.Frame(root)
    button_frame.pack(pady=5)

    upload_btn = tk.Button(button_frame, text="📁 Upload Image", command=upload_image, width=20)
    upload_btn.grid(row=0, column=0, padx=10, pady=5)

    preprocess_btn = tk.Button(button_frame, text="⚙️ Preprocess Image", command=preprocess_image, width=20)
    preprocess_btn.grid(row=0, column=1, padx=10, pady=5)

    analyze_btn = tk.Button(button_frame, text="📏 Analyze Size", command=analyze_size, width=20)
    analyze_btn.grid(row=1, column=0, padx=10, pady=5)

    conc_btn = tk.Button(button_frame, text="💧 Compute Concentration", command=compute_concentration, width=20)
    conc_btn.grid(row=1, column=1, padx=10, pady=5)

    export_btn = tk.Button(button_frame, text="📄 Export Report", command=export_report, width=20)
    export_btn.grid(row=2, column=0, padx=10, pady=5)

    # rám s obrázkami
    image_frame = tk.Frame(root)
    image_frame.pack(pady=20)

    tk.Label(image_frame, text="Pôvodný obrázok").grid(row=0, column=0, padx=30)
    tk.Label(image_frame, text="Spracovaný obrázok").grid(row=0, column=1, padx=30)

    original_label = tk.Label(image_frame)
    original_label.grid(row=1, column=0)
    processed_label = tk.Label(image_frame)
    processed_label.grid(row=1, column=1)

    root.mainloop()


if __name__ == "__main__":
    main()
