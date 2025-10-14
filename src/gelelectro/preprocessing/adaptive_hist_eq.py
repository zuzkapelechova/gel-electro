import os
import matplotlib.pyplot as plt
from skimage import io, exposure
from skimage.util import img_as_ubyte
from skimage.restoration import denoise_tv_chambolle



def adaptive_hist_eq(img, clip_limit=0.02, nbins=256, visualize=False):
    """
    Aplikuje adaptívnu histogramovú ekvalizáciu (CLAHE) na zlepšenie lokálneho kontrastu.
    
    Parametre:
    - img: 2D numpy array (grayscale)
    - clip_limit: limit pre CLAHE (nižšie hodnoty = menej halo)
    - nbins: počet binov histogramu
    - visualize: bool, či zobraziť pred a po
    """
    img_eq = exposure.equalize_adapthist(img, clip_limit=clip_limit, nbins=nbins)

    if visualize:
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.imshow(img, cmap='gray')
        plt.title("Original")
        plt.axis('off')

        plt.subplot(1,2,2)
        plt.imshow(img_eq, cmap='gray')
        plt.title("After CLAHE")
        plt.axis('off')
        plt.show()
    
    return img_eq

def main(input_path, output_path, visualize=True):
    if not os.path.exists(input_path):
        print(f"File {input_path} not found!")
        return

    # Načíta obrázok v grayscale
    img = io.imread(input_path, as_gray=True)

    # Použije CLAHE
    img_eq = adaptive_hist_eq(img, visualize=visualize)

    # Uloží výsledok
    io.imsave(output_path, img_as_ubyte(img_eq))
    print(f"Saved enhanced image to {output_path}")


if __name__ == "__main__":
    input_file = "data/processed/tv_sample01.jpg"  # výstup z predchádzajúceho kroku
    output_file = "data/processed/hist_eq_sample01.jpg"   # vylepšený obrázok
    main(input_file, output_file, visualize=True)
