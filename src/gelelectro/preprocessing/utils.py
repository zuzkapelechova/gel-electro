import numpy as np
from skimage import morphology, exposure
from skimage.util import img_as_float

class GelPreprocessor:
    """ Trieda na štandardizáciu obrázkov """

    # preklápa obrázky, aby boli všetky biele na čiernom
    @staticmethod
    def detect_and_invert(img, low_perc=5, high_perc=95):
        img = img_as_float(img)
        p_low, p_high = np.percentile(img, [low_perc, high_perc])

        if p_high > 0.5:
            img = 1 - img
        return img

    # rolling ball metóda: adaptívne odčítanie pozadia
    @staticmethod
    def adaptive_bg_subtraction(img: np.ndarray):
        mean_intensity = np.mean(img)
        std_intensity = np.std(img)
        contrast = std_intensity / (mean_intensity + 1e-5)

        radius = int(np.clip(200 * np.exp(-4 * contrast), 20, 180))
        selem = morphology.disk(radius)
        background = morphology.opening(img, selem)

        img_sub = img - background
        img_sub = exposure.rescale_intensity(img_sub, in_range="image", out_range=(0, 1))
        return img_sub

    # jednoduchý threshold podľa percentilu
    @staticmethod
    def threshold_01(img, percentile):
        thresh = np.percentile(img, percentile)
        mask = img > thresh
        return mask

    # izoluje najjasnejšie pásy
    @staticmethod
    def isolate_bright_bands(img, percentile=90):
        img_norm = exposure.rescale_intensity(img, in_range="image", out_range=(0, 1))
        mask = img_norm > np.percentile(img_norm, percentile)
        return mask, img_norm

    # čistí masku eróziou a dilatáciou
    @staticmethod
    def clean_mask(mask, erosion_radius=1, dilation_radius=1):
        mask_eroded = morphology.erosion(mask, morphology.disk(erosion_radius))
        mask_cleaned = morphology.dilation(mask_eroded, morphology.disk(dilation_radius))
        return mask_cleaned

    # dvojstupňové thresholdovanie
    @classmethod
    def threshold_02(cls, img, high_perc=95, low_perc=70, erosion_radius=1, dilation_radius=1):
        bright_mask, img_norm = cls.isolate_bright_bands(img, percentile=high_perc)
        clean_bright_mask = cls.clean_mask(bright_mask, erosion_radius, dilation_radius)
        final_mask = cls.threshold_01(img_norm, percentile=low_perc)

        combined_mask = clean_bright_mask & final_mask
        result = img * combined_mask
        return result

    # celá pipeline naraz
    @classmethod
    def process_image(cls, img, inver_low=5, inver_high=95,
                      high_perc=95, low_perc=70, erosion_radius=1, dilation_radius=1):
        img_inv = cls.detect_and_invert(img, inver_low, inver_high)
        img_bg = cls.adaptive_bg_subtraction(img_inv)
        img_thresh = cls.threshold_02(img_bg, high_perc, low_perc, erosion_radius, dilation_radius)
        return img_thresh