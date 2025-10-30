from skimage import io
from skimage.util import img_as_ubyte
from skimage.transform import resize
from utils import GelPreprocessor  # relatívny import, trieda v rovnakom priečinku

# 1️⃣ načítanie obrázka
img = io.imread("data/test_samples/sample1.png", as_gray=True)

# 2️⃣ zmenšenie obrázka (ak je veľmi veľký)
#img = resize(img, (img.shape[0] // 2, img.shape[1] // 2), anti_aliasing=True)

# 3️⃣ spracovanie cez pipeline
processed_img = GelPreprocessor.process_image(img)

# 4️⃣ uloženie výsledku
io.imsave("data/processed/sample1_processed.png", img_as_ubyte(processed_img))
print("Processed image saved!")