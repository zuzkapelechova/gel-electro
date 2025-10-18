import numpy as np
from skimage import io
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt

def firefly_threshold(image, n_fireflies=20, n_iter=50, alpha=0.5, beta=0.2, gamma=1.0):
    """
    Firefly Algorithm thresholding pre svetlé pásy na tmavom pozadí.

    image : np.ndarray (float [0,1])
    n_fireflies : počet kandidátov
    n_iter : počet iterácií
    alpha, beta, gamma : parametre FA
    """
    image = image.astype(float)

    # inicializácia thresholdov náhodne medzi min a max obrázka
    T_min, T_max = np.min(image), np.max(image)
    fireflies = np.random.uniform(T_min, T_max, size=n_fireflies)

    def brightness(T):
        mask_bg = image < T
        if np.any(mask_bg):
            return -np.mean(image[mask_bg])  # chceme minimalizovať jas pozadia
        else:
            return -1e6  # penalizácia prázdnej masky

    for _ in range(n_iter):
        for i in range(n_fireflies):
            for j in range(n_fireflies):
                if brightness(fireflies[j]) > brightness(fireflies[i]):
                    r = abs(fireflies[i] - fireflies[j])
                    attraction = beta * np.exp(-gamma * r**2)
                    fireflies[i] += attraction * (fireflies[j] - fireflies[i]) + alpha * (np.random.rand()-0.5)

        # obmedzenie thresholdov na validny interval
        fireflies = np.clip(fireflies, T_min, T_max)

    # vyber najlepší threshold
    T_opt = fireflies[np.argmax([brightness(f) for f in fireflies])]
    mask = image >= T_opt

    return mask, T_opt

def firefly(image_path):
    img = io.imread(image_path, as_gray=True)
    mask, T = firefly_threshold(img / 255.0)
    overlay = np.where(mask, img, 0)
    return overlay

if __name__ == "__main__":
    test_image_path = "data/test_samples/sample02_processed.jpg"
    processed_img = firefly(io.imread(test_image_path, as_gray=True))
    io.imsave("data/processed/sample02_firefly.jpg", img_as_ubyte(processed_img))