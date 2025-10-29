from skimage import io
from skimage.util import img_as_ubyte
from utils import GelPreprocessor  # import tvojej triedy

img = io.imread("data/test_samples/sample8.png", as_gray=True)
preprocessor = GelPreprocessor()
processed_img = preprocessor.process_image(img)

io.imsave("data/test_samples/sample8_edit.png", img_as_ubyte(processed_img))