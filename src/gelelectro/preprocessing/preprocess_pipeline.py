from skimage import io
from skimage.util import img_as_ubyte
from skimage.transform import resize
from utils import GelPreprocessor  # trieda je v rovnakom priečinku

# načítanie obrázka
img = io.imread("/home/n_svobodnik/2025-1002-BF_test_leuB_2.jpg", as_gray=True)

# spracovanie cez novú pipeline
processed_img = GelPreprocessor.process_image(img)

# uloženie výsledku
io.imsave("/home/n_svobodnik/img.png", img_as_ubyte(processed_img))
print("Processed image saved!")