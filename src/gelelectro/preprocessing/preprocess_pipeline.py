# test_pipeline_new.py
from skimage import io
from skimage.util import img_as_ubyte
from skimage.transform import resize
from utils2 import GelPreprocessor2  # trieda je v rovnakom priečinku

# 1️⃣ načítanie obrázka
img = io.imread("data/test_samples/sample8.png", as_gray=True)

# 2️⃣ voliteľné zmenšenie obrázka, ak je veľmi veľký
# img = resize(img, (img.shape[0] // 2, img.shape[1] // 2), anti_aliasing=True)

# 3️⃣ spracovanie cez novú pipeline
processed_img = GelPreprocessor2.process_image(img)

# 4️⃣ uloženie výsledku
io.imsave("data/processed/sample8_02.png", img_as_ubyte(processed_img))
print("Processed image saved!")