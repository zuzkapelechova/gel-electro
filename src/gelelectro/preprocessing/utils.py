import numpy as np
from skimage import morphology, exposure
from skimage.util import img_as_float32
from skimage.color import rgb2gray

class GelPreprocessor:
    """Trieda na štandardizáciu gélových obrázkov"""

    @staticmethod
    def detect_and_invert(img, low_perc=5, high_perc=95):
        """Preklopí obraz, aby boli objekty svetlé na tmavom pozadí"""
        img = img_as_float32(img)
        p_low, p_high = np.percentile(img, [low_perc, high_perc])
        if np.mean(img) > 0.5:  # adaptívnejšie ako len p_high
            img = 1 - img
        return img

    @staticmethod
    def adaptive_bg_subtraction(img: np.ndarray):
        """Odstráni pozadie adaptívne pomocou morfologického openingu"""
        mean_intensity = np.mean(img)
        std_intensity = np.std(img)
        contrast = std_intensity / (mean_intensity + 1e-5)

        radius = int(np.clip(200 * np.exp(-4 * contrast), 20, 180))
        selem = morphology.disk(radius)
        background = morphology.opening(img, selem)

        img_sub = img - background
        img_sub = exposure.rescale_intensity(img_sub, in_range="image", out_range=(0, 1))
        return img_sub

    @staticmethod
    def threshold_01(img, percentile):
        """Jednoduché percentilové prahovanie"""
        thresh = np.percentile(img, percentile)
        return img > thresh

    @staticmethod
    def isolate_bright_bands(img, percentile=90):
        """Izoluje najjasnejšie oblasti a normalizuje obraz"""
        img_norm = exposure.rescale_intensity(img, in_range="image", out_range=(0, 1))
        mask = img_norm > np.percentile(img_norm, percentile)
        return mask, img_norm

    @staticmethod
    def clean_mask(mask, radius=1):
        """Vyčistí masku morfologickým openingom (erózia + dilatácia v jednom kroku)"""
        return morphology.opening(mask, morphology.disk(radius))

    @classmethod
    def threshold_02(cls, img, high_perc=95, low_perc=70, mask_radius=1):
        bright_mask, img_norm = cls.isolate_bright_bands(img, percentile=high_perc)
        clean_bright_mask = cls.clean_mask(bright_mask, radius=mask_radius)
        final_mask = cls.threshold_01(img_norm, percentile=low_perc)

        combined_mask = clean_bright_mask & final_mask
        return img * combined_mask

    @classmethod
    def process_image(cls, img, inver_low=5, inver_high=95,
                      high_perc=95, low_perc=70, mask_radius=1):
        """Celá pipeline naraz: invert → background → threshold"""
        if img.shape[-1] == 4:  # RGBA → RGB
            img = img[..., :3]
        if img.ndim == 3:  # RGB
            img = rgb2gray(img)
        img = img_as_float32(img)

        img_inv = cls.detect_and_invert(img, inver_low, inver_high)
        img_bg = cls.adaptive_bg_subtraction(img_inv)
        img_thresh = cls.threshold_02(img_bg, high_perc, low_perc, mask_radius=mask_radius)

        return img_thresh
