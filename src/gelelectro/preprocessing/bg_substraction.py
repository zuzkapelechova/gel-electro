import numpy as np
from skimage import io
from skimage.util import img_as_ubyte, img_as_float
from scipy.ndimage import gaussian_filter

def subtract_background(image, radius_x=30, radius_y=30, scale=True):
    """
    Approximácia logiky pozadia z GelImage:
    - Odpočíta pomaly sa meniace pozadie (Gaussian blur)
    - Pri 'scale=True' škáluje výsledok do rozsahu 0–1
    """
    # Gaussian filter pre pozadie (mierne rozmazané)
    background = gaussian_filter(image, sigma=(radius_y, radius_x))
    
    # Odpočítanie pozadia
    subtracted = image - background
    subtracted = np.clip(subtracted, 0, None)

    if scale:
        subtracted = subtracted / np.max(subtracted)

    return subtracted

if __name__ == "__main__":
    input_path = "data/processed/bg_sample04.jpg"
    output_path = "data/processed/substracted_sample04.jpg"

    image = img_as_float(io.imread(input_path, as_gray=True))

    # Otestuj s viacerými parametrami
    image_sub = subtract_background(image, radius_x=25, radius_y=25, scale=True)

    io.imsave(output_path, img_as_ubyte(image_sub))
    print("✅ Background subtraction done and saved to:", output_path)