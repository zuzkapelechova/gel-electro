import numpy as np
from skimage import io, filters, exposure, morphology
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt

def threshold_mask(img, percentile):
    thresh = np.percentile(img, percentile)
    mask = img > thresh
    return mask

def isolate_bright_bands(image, percentile=90):
    """
    Zachová len najjasnejšie (biele) časti obrázka.
    percentile určuje, aké percento najjasnejších pixelov zostane (napr. 90 = top 10 %).
    """
    # 1️⃣ normalizácia na 0–1
    img_norm = exposure.rescale_intensity(image, in_range="image", out_range=(0, 1))

    # 2️⃣ nájdenie prahu podľa percentilu jasu
    thresh = np.percentile(img_norm, percentile)

    # 3️⃣ všetko tmavšie než threshold (teda sivé + čierne) ide preč
    mask = img_norm > thresh

    return mask, img_norm

def clean_mask(mask: np.ndarray, erosion_radius=1, dilation_radius=1):
    """
    Odstráni drobné artefakty z binárnej masky pomocou erózie a dilatácie.

    Parameters:
        mask (np.ndarray): Binárna maska (True/False alebo 0/1)
        erosion_radius (int): Polomer štruktúrneho elementu pre eróziu
        dilation_radius (int): Polomer štruktúrneho elementu pre dilatáciu

    Returns:
        np.ndarray: Upravená binárna maska
    """
    # Štruktúrny element pre eróziu
    selem_ero = morphology.disk(erosion_radius)
    mask_eroded = morphology.erosion(mask, selem_ero)

    # Štruktúrny element pre dilatáciu
    selem_dil = morphology.disk(dilation_radius)
    mask_cleaned = morphology.dilation(mask_eroded, selem_dil)

    return mask_cleaned

def threshold(image: np.ndarray, high_perc=95, low_perc=70, erosion_radius=1, dilation_radius=1):
    """
    Dvojstupňové thresholdovanie:
    1. Zachová len najjasnejšie časti obrázka podľa high_perc.
    2. Vyčistí masku pomocou erózie a dilatácie.
    3. Aplikácia druhej masky podľa low_perc na pôvodný obrázok.

    Parameters:
        image (np.ndarray): Vstupný obrázok
        high_perc (float): Percentil pre prvé thresholdovanie
        low_perc (float): Percentil pre druhé thresholdovanie
        erosion_radius (int): Polomer pre eróziu
        dilation_radius (int): Polomer pre dilatáciu

    Returns:
        np.ndarray: Výsledný obrázok po dvojstupňovom thresholdovaní
    """
    # 1️⃣ Izolácia najjasnejších pásov
    bright_mask, img_norm = isolate_bright_bands(image, percentile=high_perc)

    # 2️⃣ Čistenie masky
    clean_bright_mask = clean_mask(bright_mask, erosion_radius, dilation_radius)

    # 3️⃣ Druhé thresholdovanie na pôvodný obrázok
    final_mask = threshold_mask(img_norm, percentile=low_perc)

    # Kombinácia masiek
    combined_mask = clean_bright_mask & final_mask

    # Aplikácia masky na pôvodný obrázok
    result = image * combined_mask

    return result



if __name__ == "__main__":
    test_image_path = "data/test_samples/sample03_processed.jpg"
    processed_img = threshold(io.imread(test_image_path, as_gray=True))
    io.imsave("data/processed/sample03_thresholded.jpg", img_as_ubyte(processed_img))
