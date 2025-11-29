import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from skimage import io
import numpy as np
from utils import GelPreprocessor, compute_sample_sizes, align_img, generate_report, get_sample_concentrations, get_band_weights


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
    preprocess_results_text = None
    annotated_size_img = None
    size_results_text = None
    annotated_conc_img = None
    conc_results_text = None
    annotated_weight_img = None
    weight_results_text = None

    # stores the sample volume (in microliters), entered by user
    # you can access its value using sample_volume.get()
    sample_volume = tk.DoubleVar(value=False)


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
        nonlocal processed_img, preprocess_results_text
        if original_img is None:
            messagebox.showwarning("Warning", "Please upload an image first!")
            return
        
        # get number of ladders from Radiobutton
        ladder2_flag = True if ladder_var.get() == 2 else False

        processed_img, results = preprocessor.process_image(original_img)
        preprocess_results_text = results

        if ladder2_flag:
            aligned = align_img(processed_img)
            processed_img = aligned

        # show processed image
        show_image(processed_img, processed_label)
        messagebox.showinfo("Done", "Image has been processed!")

    # analyze band sizes
    def analyze_size():
        nonlocal annotated_size_img, size_results_text
        if processed_img is None:
            messagebox.showwarning("Warning", "Please preprocess the image first!")
            return

        # get number of ladders from Radiobutton
        ladder2_flag = True if ladder_var.get() == 2 else False

        # call function from utils
        annotated_size_img, size_results_text = compute_sample_sizes(processed_img, ladder2=ladder2_flag)

        # show results on the image
        show_image(annotated_size_img, processed_label)
        messagebox.showinfo("Done", "Image has been analysed!")

    # concentration analysis
    def compute_concentration():
        nonlocal annotated_conc_img, conc_results_text

        if processed_img is None:
            messagebox.showwarning("Warning", "Please preprocess the image first!")
            return

        vol = sample_volume.get()
        if vol == 0:
            messagebox.showerror(
                "Input Error",
                "Please enter sample volume before running concentration analysis."
            )
            return

        # get number of ladders from Radiobutton
        ladder2_flag = True if ladder_var.get() == 2 else False

        # call function from utils
        annotated_conc_img, conc_results_text = get_sample_concentrations(processed_img, 10, volume=sample_volume.get(), ladder2=ladder2_flag)

        # show results on the image
        show_image(annotated_conc_img, processed_label)
        messagebox.showinfo("Done", "Image has been analysed!")

    # band weight estimation
    def estimate_band_weights():
        nonlocal annotated_weight_img, weight_results_text

        if processed_img is None:
            messagebox.showwarning("Warning", "Please preprocess the image first!")
            return

        # get number of ladders from Radiobutton
        ladder2_flag = True if ladder_var.get() == 2 else False

        # call function from utils
        annotated_weight_img, weight_results_text = get_band_weights(processed_img, ladder2=ladder2_flag)

        # show results on the image
        show_image(annotated_weight_img, processed_label)
        messagebox.showinfo("Done", "Image has been analysed!")

    # export report
    def export_report():
        if original_img is None:
            messagebox.showwarning("Warning", "Please upload an image first!")
            return

        if processed_img is None:
            messagebox.showwarning("Warning", "Please preprocess the image first!")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[("HTML Report", "*.html")]
        )
        if not file_path:
            return

        generate_report(
            output_html_path=file_path,
            preprocess_params=preprocess_results_text,
            original_img=original_img,
            preprocessed_img=processed_img,
            annotated_size_img=annotated_size_img,
            size_results_text=size_results_text,
            annotated_conc_img=annotated_conc_img,
            conc_results_text=conc_results_text,
            annotated_weight_img=annotated_weight_img,
            weight_results_text=weight_results_text,
            sample_volume=sample_volume.get()
        )

        messagebox.showinfo("Success", "Report exported successfully!")

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

    # info button
    def show_info():
        messagebox.showinfo(
            "About / Help",
            "HOW TO USE THE GEL ANALYSIS TOOL:\n\n"
            "1. Upload an image of a gel (.jpg, .png, .tif).\n"
            "2. Click 'Preprocess Image' to prepare it for analysis.\n"
            "3. Use 'Analyze Size' to detect DNA band sizes.\n"
            "4. To compute sample concentration, enter the sample volume (in µL) and click 'Compute Concentration'.\n"
            "5. To estimate band weights, click 'Estimate Band Weights'.\n"
            "6. Export the full analysis as an HTML report.\n\n"

            "CONDITIONS FOR CORRECT PROCESSING:\n\n"
            "1. Ladder must be the FIRST lane on the left (or choose '2 ladders' if ladders are on both sides).\n"
            "2. The gel image should be as horizontally aligned as possible.\n"
            "3. Crop the top of the gel exactly at the wells (start of the lanes).\n"
            "4. Only ONE gel should be present in the uploaded image.\n"
            "5. For concentration analysis, you must enter a non-zero sample volume.\n\n"
        )

    info_button = tk.Button(root, text="i", font=("Arial", 16, "bold"),
                            command=show_info)
    info_button.place(relx=0.95, rely=0.05, anchor="ne")


    # buttons frame
    button_frame = tk.Frame(root)
    button_frame.pack(pady=5)

    upload_btn = tk.Button(button_frame, text="📁 Upload Image", command=upload_image, width=20)
    upload_btn.grid(row=0, column=0, padx=10, pady=5)

    preprocess_btn = tk.Button(button_frame, text="⚙️ Preprocess Image", command=preprocess_image, width=20)
    preprocess_btn.grid(row=0, column=1, padx=10, pady=5)

    analyze_btn = tk.Button(button_frame, text="📏 Analyse Size", command=analyze_size, width=20)
    analyze_btn.grid(row=1, column=0, padx=10, pady=5)

    conc_btn = tk.Button(button_frame, text="💧 Compute Concentration", command=compute_concentration, width=20)
    conc_btn.grid(row=1, column=1, padx=10, pady=5)

    weight_btn = tk.Button(button_frame, text="Estimate Band Weights", command=estimate_band_weights, width=20)
    weight_btn.grid(row=2, column=0, padx=10, pady=5)

    export_btn = tk.Button(button_frame, text="📄 Export Report", command=export_report, width=20)
    export_btn.grid(row=3, column=0, columnspan=2, padx=10, pady=5)

    # ladder count selection
    ladder_var = tk.IntVar(value=1)
    ladder_frame = tk.Frame(button_frame)
    ladder_frame.grid(row=4, column=0, columnspan=2, pady=10)

    tk.Label(ladder_frame, text="Ladders:").pack(side=tk.LEFT, padx=5)
    tk.Radiobutton(ladder_frame, text="1 Ladder", variable=ladder_var, value=1).pack(side=tk.LEFT)
    tk.Radiobutton(ladder_frame, text="2 Ladders", variable=ladder_var, value=2).pack(side=tk.LEFT)

    # input field to enter sample volume in microliters
    volume_frame = tk.Frame(button_frame)
    volume_frame.grid(row=2, column=1, pady=10)

    tk.Label(volume_frame, text="Sample volume (µl):").pack(side=tk.LEFT, padx=5)
    volume_entry = tk.Entry(volume_frame, textvariable=sample_volume, width=8)
    volume_entry.pack(side=tk.LEFT)

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
