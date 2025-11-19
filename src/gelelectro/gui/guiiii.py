import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from skimage import io
import numpy as np
from utils import GelPreprocessor, compute_sample_sizes


def main():
    root = tk.Tk()
    root.title("Gel Analysis Tool")

    # full screen window
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.geometry(f"{screen_width}x{screen_height}")

    preprocessor = GelPreprocessor()

    # image storage
    original_img = None
    processed_img = None

    # ====== FUNCTIONS ======

    # upload image
    def upload_image():
        nonlocal original_img
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.png *.tif *.tiff")]
        )
        if not file_path:
            return
        original_img = io.imread(file_path, as_gray=True)
        show_image(original_img, original_label)
        messagebox.showinfo("Image", "Image uploaded successfully!")

    # preprocess image
    def preprocess_image():
        nonlocal processed_img
        if original_img is None:
            messagebox.showwarning("Warning", "Please upload an image first!")
            return
        
        processed_img, results = preprocessor.process_image(original_img)

        # show processed image
        show_image(processed_img, processed_label)

        # scrollable window with results
        result_window = tk.Toplevel(root)
        result_window.title("Preprocessing Results")

        text_widget = tk.Text(result_window, wrap=tk.NONE, width=80, height=30)
        text_widget.insert(tk.END, results)
        text_widget.config(state=tk.DISABLED)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(result_window, command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.config(yscrollcommand=scrollbar.set)

        messagebox.showinfo("Done", "Image has been processed!")

    # analyze band sizes
    def analyze_size():
        if processed_img is None:
            messagebox.showwarning("Warning", "Please preprocess the image first!")
            return

        # get number of ladders from Radiobutton
        ladder2_flag = True if ladder_var.get() == 2 else False

        # call function from utils
        annotated, results = compute_sample_sizes(processed_img, ladder2=ladder2_flag)

        # show results on the image
        show_image(annotated, processed_label)

        # scrollable window with results
        result_window = tk.Toplevel(root)
        result_window.title("Analysis Results")

        text_widget = tk.Text(result_window, wrap=tk.NONE, width=80, height=30)
        text_widget.insert(tk.END, results)
        text_widget.config(state=tk.DISABLED)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(result_window, command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.config(yscrollcommand=scrollbar.set)

    # placeholder – concentration analysis
    def compute_concentration():
        if processed_img is None:
            messagebox.showwarning("Warning", "Please preprocess the image first!")
            return
        messagebox.showinfo(
            "Concentration Analysis",
            "This function is not implemented yet."
        )

    # placeholder – export report
    def export_report():
        messagebox.showinfo(
            "Export",
            "This function is not implemented yet."
        )

    # display image while keeping aspect ratio
    def show_image(img, label_widget, max_width=600, max_height=600):
        if np.issubdtype(img.dtype, np.floating):
            img_display = (img * 255).astype(np.uint8)
        else:
            img_display = img.copy()

        img_pil = Image.fromarray(img_display)

        original_width, original_height = img_pil.size
        ratio = min(max_width / original_width, max_height / original_height)
        new_size = (int(original_width * ratio), int(original_height * ratio))
        img_pil = img_pil.resize(new_size, Image.Resampling.LANCZOS)

        img_tk = ImageTk.PhotoImage(img_pil)
        label_widget.config(image=img_tk)
        label_widget.image = img_tk

    # layout GUI
    title = tk.Label(root,
                     text="Electrophoretic Gel Analysis",
                     font=("Arial", 20, "bold"))
    title.pack(pady=30)

    # buttons frame
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

    # ladder count selection
    ladder_var = tk.IntVar(value=1)
    ladder_frame = tk.Frame(button_frame)
    ladder_frame.grid(row=3, column=0, columnspan=2, pady=10)

    tk.Label(ladder_frame, text="Ladders:").pack(side=tk.LEFT, padx=5)
    tk.Radiobutton(ladder_frame, text="1 Ladder", variable=ladder_var, value=1).pack(side=tk.LEFT)
    tk.Radiobutton(ladder_frame, text="2 Ladders", variable=ladder_var, value=2).pack(side=tk.LEFT)

    # image display frame
    image_frame = tk.Frame(root)
    image_frame.pack(pady=20)

    tk.Label(image_frame, text="Unprocessed Image").grid(row=0, column=0, padx=30)
    tk.Label(image_frame, text="Processed Image").grid(row=0, column=1, padx=30)

    original_label = tk.Label(image_frame)
    original_label.grid(row=1, column=0)
    processed_label = tk.Label(image_frame)
    processed_label.grid(row=1, column=1)

    root.mainloop()


if __name__ == "__main__":
    main()
