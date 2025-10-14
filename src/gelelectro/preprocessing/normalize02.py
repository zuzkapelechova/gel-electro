import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure
from skimage.restoration import denoise_tv_chambolle
from skimage.morphology import opening, disk
from skimage.filters import threshold_multiotsu
from skimage.util import img_as_float, img_as_ubyte


def base_normalize_fixed(img, gel_mean=0.7, background_mean=0.3, low_clip=0.01, high_clip=0.99):
    """
    Nastaví pevný cieľový jas pre gél a pozadie.
    - gel_mean: cieľová intenzita pre pásy/gél
    - background_mean: cieľová intenzita pre pozadie
    """
    img = img_as_float(img)

    # 1️⃣ orez extrémnych hodnôt (odstráni saturácie)
    p_low, p_high = np.percentile(img, (low_clip*100, high_clip*100))
    img = np.clip((img - p_low) / (p_high - p_low), 0, 1)

    # 2️⃣ lineárne škálovanie podľa pevného pozadia a gelu
    min_val = np.min(img)
    max_val = np.max(img)
    # lineárne premapovanie do nového rozsahu
    img = (img - min_val) / (max_val - min_val)  # 0-1
    img = img * (gel_mean - background_mean) + background_mean
    img = np.clip(img, 0, 1)

    return img

def detect_and_correct_polarity(img, visualize=False):
    """Zistí, či je obrázok invertovaný (svetlé pozadie, tmavé pásy) a ak áno, obráti ho."""
    mean_intensity = np.mean(img)

    # ak je obrázok veľmi svetlý, invertujeme ho
    inverted = False
    if mean_intensity > 0.5:
        img = 1 - img
        inverted = True

    if visualize:
        print(f"Mean intensity: {mean_intensity:.3f}, inverted: {inverted}")

    return img, inverted

def normalize_piecewise_adaptive(img, clip_limit=0.02):
    # adaptívna hist. equalizácia – rozdeľ histogram do regiónov
    return exposure.equalize_adapthist(img, clip_limit=clip_limit)

def nonlinear_diffusion_filter(img, weight=0.05):
    # tu použijeme TV ako aproximačnú náhradu za difúziu
    return denoise_tv_chambolle(img, weight=weight)

def background_correction_threshold(img):
    # použijeme viacúrovňové prahovanie (multi-Otsu) na odhad pozadia
    thresholds = threshold_multiotsu(img, classes=3)
    # thresholds získa hranice, pozadie je pred prvým prahom
    mask = img <= thresholds[0]
    background_value = np.mean(img[mask])
    corrected = img - background_value
    corrected = np.clip(corrected, 0, 1)
    return corrected

def pipeline_article_style(img_path):
    img = img_as_float(io.imread(img_path, as_gray=True))

    # 0️⃣ zjednotenie expozície
    img_base = base_normalize_fixed(img, gel_mean=0.7, background_mean=0.3)
    
    # 1️⃣ detekcia polarity (ak bude treba)
    img, inverted = detect_and_correct_polarity(img)

    # 1️⃣ normalizácia
    img_norm = normalize_piecewise_adaptive(img, clip_limit=0.03)

    # 2️⃣ filtrácia
    img_filt = nonlinear_diffusion_filter(img_norm, weight=0.03)

    # 3️⃣ korekcia pozadia
    img_bgcorr = background_correction_threshold(img_filt)

    return img, img_norm, img_filt, img_bgcorr

def plot_steps(imgs, titles):
    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    for ax, im, title in zip(axes, imgs, titles):
        ax.imshow(im, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    inp = "data/test_samples/sample01.jpg"
    orig, norm, filt, bgcorr = pipeline_article_style(inp)
    plot_steps([orig, norm, filt, bgcorr],
               ["Original", "Normalized (piecewise)", "Filtered (TV)", "BG Corrected"])