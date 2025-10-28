from skimage import io
from skimage.util import img_as_ubyte
from utils import GelPreprocessor  # relatívny import, trieda v rovnakom priečinku

# 1️⃣ načítanie obrázka
img = io.imread("data/test_samples/sample01_processed.jpg", as_gray=True)

# 2️⃣ spracovanie cez pipeline
processed_img = GelPreprocessor.process_image(img)

# 3️⃣ uloženie výsledku
io.imsave("data/processed/sample01_processed.jpg", img_as_ubyte(processed_img))
print("Processed image saved!")