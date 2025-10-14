import os
from skimage import io
from skimage.util import img_as_float, img_as_ubyte
from skimage.filters import median
from skimage.morphology import disk
from skimage.restoration import denoise_tv_chambolle
import matplotlib.pyplot as plt

def enhanced_denoise(img, median_disk=2, tv_weight=0.05, visualize=True):
    """
    Nonlinear filtering pre invertované a normalizované gelové obrázky.
    Kombinuje median filter + Total Variation denoising.

    Parametre:
    - img: float obrázok v rozsahu [0,1]
    - median_disk: veľkosť okna pre median filter (disk)
    - tv_weight: weight pre TV denoising
    - visualize: či zobraziť pôvodný a spracovaný obrázok
    
    Návrat:
    - img_filtered: spracovaný obrázok
    """
    
    # Median filter na drobný šum a artefakty
    img_med = median(img, disk(median_disk))
    
    # Total Variation na hladké pozadie, menšie weight = menej rozmazania pásov
    img_tv = denoise_tv_chambolle(img_med, weight=tv_weight)
    
    if visualize:
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.imshow(img, cmap='gray')
        plt.title("Original (normalized & inverted)")
        plt.axis('off')
        
        plt.subplot(1,2,2)
        plt.imshow(img_tv, cmap='gray')
        plt.title("After enhanced nonlinear filtering")
        plt.axis('off')
        plt.show()
    
    return img_tv

# ------------------- TESTOVANIE -------------------

if __name__ == "__main__":
    input_path = "data/processed/bg_sample03.jpg"  # nahraď vlastnou cestou
    output_path = "data/processed/tv_sample03.jpg"

    if not os.path.exists(input_path):
        print(f"File {input_path} not found!")
        exit(1)

    # Načítanie obrázku
    img = img_as_float(io.imread(input_path, as_gray=True))

    # Aplikovanie enhanced denoising
    img_filtered = enhanced_denoise(img, median_disk=2, tv_weight=0.05, visualize=True)

    # Uloženie výsledku
    io.imsave(output_path, img_as_ubyte(img_filtered))
    print(f"Processed image saved to {output_path}")
