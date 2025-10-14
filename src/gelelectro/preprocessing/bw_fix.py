import numpy as np
from skimage import io, img_as_float, img_as_ubyte
import matplotlib.pyplot as plt
import os

def detect_and_invert(img, low_perc=5, high_perc=95, visualize=False):
    img = img_as_float(img)
    
    p_low, p_high = np.percentile(img, [low_perc, high_perc])
    
    inverted = False
    # ak je pozadie svetlé, invertujeme
    if p_high > 0.5:
        img = 1 - img
        inverted = True

    if visualize:
        print(f"Percentil {low_perc}%: {p_low:.3f}, {high_perc}%: {p_high:.3f}, inverted: {inverted}")
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.imshow(img, cmap='gray')
        plt.title("Inverted" if inverted else "Original")
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.hist(img.ravel(), bins=256)
        plt.title("Histogram")
        plt.show()
        
    return img, inverted

def normalize_fixed(img, background_mean=0.3, gel_mean=0.7, visualize=False):
    img = img_as_float(img)
    p_low, p_high = np.percentile(img, [5, 95])
    scale = (gel_mean - background_mean) / (p_high - p_low + 1e-8)
    img_norm = (img - p_low) * scale + background_mean
    img_norm = np.clip(img_norm, 0, 1)
    if visualize:
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.imshow(img, cmap='gray')
        plt.title("After Invert")
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(img_norm, cmap='gray')
        plt.title("Normalized Fixed")
        plt.axis('off')
        plt.show()
    return img_norm

def main(input_path, output_path, visualize=True):
    if not os.path.exists(input_path):
        print(f"File {input_path} not found!")
        return
    
    # 1️⃣ Invert podľa polarity
    img = io.imread(input_path, as_gray=True)
    img_inverted, _ = detect_and_invert(img, visualize=visualize)
    
    # 2️⃣ Fixná normalizácia intenzity
    img_normalized = normalize_fixed(img_inverted, visualize=visualize)
    
    # 3️⃣ Uloženie výsledku
    io.imsave(output_path, img_as_ubyte(img_normalized))
    print(f"Saved final normalized image to {output_path}")

if __name__ == "__main__":
    # nastav cestu k obrazku
    input_file = "data/test_samples/sample04.jpg"
    output_file = "data/processed/bg_sample04.jpg"
    main(input_file, output_file, visualize=True)