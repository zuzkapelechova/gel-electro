from skimage import io
from skimage.util import img_as_ubyte
from skimage.transform import resize
from utils import GelPreprocessor  # trieda je v rovnakom priečinku

# načítanie obrázka
img = io.imread("data/test_samples/sample4.png", as_gray=True)

# spracovanie cez novú pipeline
processed_img = GelPreprocessor.process_image(img)

# uloženie výsledku
io.imsave("data/processed/sample4_processed.png", img_as_ubyte(processed_img))
print("Processed image saved!")