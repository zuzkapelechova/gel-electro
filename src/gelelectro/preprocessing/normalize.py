import numpy as np
from skimage import io, color, exposure
from skimage.morphology import opening, disk
from skimage.restoration import denoise_tv_chambolle
import os

def remove_background(image, disk_size=50):
    """Odstráni nerovnomerné pozadie pomocou morfologického openingu."""
    background = opening(image, disk(disk_size))
    corrected = image - background
    # normalizácia po subtrakcii
    corrected = (corrected - corrected.min()) / (corrected.max() - corrected.min())
    return corrected

def denoise_small(image, weight=0.05):
    """Odstrani malinke artefakty (prach)"""
    return denoise_tv_chambolle(image, weight=weight)

def ensure_dark_bands(image):
    """Upraví obrázok tak, aby pásy boli tmavé a pozadie svetlé."""
    if np.mean(image) < 0.5:  # tmavé pozadie = invert
        image = 1 - image
    return image

def load_image(path):
    """Načíta obrázok z disku."""
    return io.imread(path)

def to_grayscale(image):
    """Prevedie RGB na grayscale. Ak už grayscale, vráti pôvodný obraz."""
    if len(image.shape) == 3:
        return color.rgb2gray(image)
    return image

def normalize_intensity(image):
    """Normalizuje intenzitu pixelov na rozsah 0–1."""
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def enhance_contrast(image):
    """Zlepší kontrast pomocou CLAHE (adaptive histogram equalization)."""
    return exposure.equalize_adapthist(image)

def normalize_image(img_path, save_path=None):
    img = load_image(img_path)
    img = to_grayscale(img)
    img = ensure_dark_bands(img)
    img = normalize_intensity(img)
    #img = remove_background(img)
    img = enhance_contrast(img)
    img = denoise_small(img, weight=0.05)

    if save_path:
        io.imsave(save_path, (img * 255).astype(np.uint8))

    return img

# Demo použitie
if __name__ == "__main__":
    input_file = "data/test_samples/sample01.jpg"
    output_file = "data/processed/sample01_normalizedIMP.jpg"
    img = normalize_image(input_file, output_file)
    print("Normalization complete, saved to", output_file)