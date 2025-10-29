import numpy as np
from skimage import io, morphology, filters
from skimage.util import img_as_ubyte, img_as_float
import matplotlib.pyplot as plt

def otsu_like_threshold(image: np.ndarray, horizontal_cleanup=True, erosion_size=3, dilation_size=5):
    """
    Threshold podľa princípu článku: všetko pod prahom je 0, ostatné ostáva.
    Môžeme aplikovať horizontálnu eróziu/dilatáciu na odstránenie šumu.
    
    image: np.ndarray, predspracovaný (tmavé pozadie, svetlé pásy)
    horizontal_cleanup: či aplikovať horizontálnu eróziu/dilatáciu
    erosion_size, dilation_size: veľkosti štruktúrneho elementu
    """
    img = img_as_float(image)
    
    # 1️⃣ Otsu threshold
    thresh = filters.threshold_otsu(img)

    # 2️⃣ vytvorenie masky: pixely > thresh ostanú, ostatné 0
    mask = img > thresh
    result = img * mask
    
    # 3️⃣ voliteľná horizontálna erózia/dilatácia
    if horizontal_cleanup:
        selem_erode = morphology.rectangle(1, erosion_size)  # horizontálne
        selem_dilate = morphology.rectangle(1, dilation_size)
        mask_clean = morphology.erosion(mask, selem_erode)
        mask_clean = morphology.dilation(mask_clean, selem_dilate)
        result = img * mask_clean
    
    return result, thresh

def otsu_threshold(image_path):
    img = io.imread(image_path, as_gray=True)
    
    # threshold + horizontal cleanup
    result, thresh = otsu_like_threshold(img)

if __name__ == "__main__":
    # 👇 zmeň na svoj testovací obrázok
    test_image_path = "data/test_samples/sample8_edit.png"
    processed_img, _ = otsu_like_threshold(io.imread(test_image_path, as_gray=True))
    
    io.imsave("data/processed/sample7_otsu_thresh.jpg", img_as_ubyte(processed_img))