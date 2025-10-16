import numpy as np
from skimage import io
from skimage.filters import threshold_local
import matplotlib.pyplot as plt

def simple_threshold_light_bands(image, block_size=51, offset=0.01):
    """
    Lokálny thresholding pre svetlé pásy na tmavom pozadí.
    block_size určuje veľkosť okna pre lokálny prah,
    offset ho mierne posúva (kladný = menej citlivé).
    """
    local_thresh = threshold_local(image, block_size=block_size, offset=offset)
    binary = image > local_thresh
    return binary

def visualize_thresholding(image_path, block_size=51, offset=0.01):
    # 1️⃣ načítanie obrázka
    img = io.imread(image_path, as_gray=True)
    
    # 2️⃣ výpočet binárnej masky
    binary = simple_threshold_light_bands(img, block_size=block_size, offset=offset)

    # 3️⃣ vytvorenie overlay (iba pásy)
    overlay = np.where(binary, img, 0)

    # 4️⃣ zobrazenie
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    titles = ["Pôvodný obrázok", "Threshold maska", "Maska na obrázku"]
    images = [img, binary, overlay]

    for ax, im, title in zip(axes, images, titles):
        ax.imshow(im, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 👇 zmeň na svoj testovací obrázok
    test_image_path = "data/test_samples/sample02_processed.jpg"

    visualize_thresholding(
        test_image_path,
        block_size=51,   # môžeš skúsiť 31, 51, 101…
        offset=0.0001      # menšie = citlivejšie, väčšie = menej
    )

