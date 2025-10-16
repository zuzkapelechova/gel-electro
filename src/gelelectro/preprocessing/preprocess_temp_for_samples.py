from skimage import io
from skimage.util import img_as_ubyte
from utils import GelPreprocessor  # import tvojej triedy

img = io.imread("data/test_samples/sample05.tif", as_gray=True)
preprocessor = GelPreprocessor()
processed_img = preprocessor.process_image(img)

io.imsave("data/processed/sample05_processed.jpg", img_as_ubyte(processed_img))