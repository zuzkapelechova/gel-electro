import numpy as np
from skimage import morphology, exposure
from skimage.util import img_as_float


class GelPreprocessor:
    """ Trieda na standardizaciu obrazkov """

    # preklapa obrazky aby boli vsetky biele na ciernom
    @staticmethod
    def detect_and_invert(img, low_perc=5, high_perc=95):
        img = img_as_float(img)
    
        p_low, p_high = np.percentile(img, [low_perc, high_perc])
    
        #inverted = False (mozeme neskor pouzit na metadata)
        if p_high > 0.5:
            img = 1 - img
            #inverted = True
        return img
    
    # rolling ball metoda (IOCBIO): adaptivne odstranovanie pozadia
    @staticmethod
    def adaptive_bg_subtraction(image: np.ndarray):
        # base stats
        mean_intensity = np.mean(image)
        std_intensity = np.std(image)
        contrast = std_intensity / (mean_intensity + 1e-5)

        # radius
        radius = int(np.clip(200 * np.exp(-4 * contrast), 20, 180))

        selem = morphology.disk(radius)
        background = morphology.opening(image, selem)

        # plati pre tmave pozadie svetle pasy
        image_sub = image - background

        # normalizacia intenzity
        image_sub = exposure.rescale_intensity(image_sub, in_range="image", out_range=(0, 1))

        return image_sub
    
    # tuto funkciu pouzivame
    def process_image(self, img: np.ndarray):
        img_inverted = self.detect_and_invert(img)
        img_processed = self.adaptive_bg_subtraction(img_inverted)
        return img_processed
