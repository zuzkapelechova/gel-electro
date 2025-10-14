import numpy as np
from skimage import filters, morphology, exposure, io
from skimage.util import img_as_float

def adaptive_background_subtraction_fixed_dark_bg(image: np.ndarray):
    """
    Automaticky určí radius pre odčítanie pozadia podľa kontrastu,
    ale berie ako samozrejmosť, že pozadie je TMAVÉ a pásy sú SVETLÉ.
    """
    # 1️⃣ základné štatistiky
    mean_intensity = np.mean(image)
    std_intensity = np.std(image)
    contrast = std_intensity / (mean_intensity + 1e-5)

    # 2️⃣ adaptívny radius podľa kontrastu
    if contrast > 0.3:
        radius = 30
    elif contrast > 0.15:
        radius = 60
    else:
        radius = 120

    # 3️⃣ výpočet pozadia cez morfologické otvorenie
    selem = morphology.disk(radius)
    background = morphology.opening(image, selem)

    # 4️⃣ odčítanie (vždy pozadie tmavé)
    image_sub = image - background

    # 5️⃣ normalizácia intenzity
    image_sub = exposure.rescale_intensity(image_sub, in_range="image", out_range=(0, 1))

    return image_sub, {
        "radius": radius,
        "contrast": contrast,
        "background_is_dark": True
    }

if __name__ == "__main__":
    img = img_as_float(io.imread("data/processed/bg_sample01.jpg", as_gray=True))
    result, params = adaptive_background_subtraction_fixed_dark_bg(img)

    print("Použité parametre:", params)
    io.imsave("data/processed/adapt_sub_sample01.jpg", (result * 255).astype(np.uint8))
