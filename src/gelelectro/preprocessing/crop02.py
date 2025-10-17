import numpy as np
from skimage import io
import matplotlib.pyplot as plt


def split_gels_horizontal_safe(img, min_gap, buffer, min_segment_height):
    """
    Horizontálne rozdelenie elektroforéz pri čiernom pozadí a bielych pásoch.
    Zachová každý gél celý, pridáva malý buffer nad a pod medzerou.
    Ignoruje malé segmenty, ktoré môžu byť okraj skenu.

    min_gap: minimalna výška medzery medzi gélmi
    buffer: počet pixelov nad a pod medzerou, ktorý sa zachová
    min_segment_height: minimálna výška segmentu, aby bol považovaný za gél
    """
    row_mean = np.mean(img, axis=1)  # priemerna hodnota pixelov v riadkoch

    # tmavé riadky = medzera
    threshold = np.percentile(row_mean, 10)
    mask_gap = row_mean <= threshold

    segments = []
    start = 0
    in_gap = False

    for i, val in enumerate(mask_gap):
        if val and not in_gap:
            in_gap = True
            gap_start = i
        elif not val and in_gap:
            in_gap = False
            if i - gap_start >= min_gap:
                end = gap_start + buffer
                segments.append((start, end))
                start = max(i - buffer, 0)

    if start < len(row_mean):
        segments.append((start, len(row_mean)))

    # orezanie podľa segmentov a filtrovanie malých segmentov
    cropped_imgs = []
    final_segments = []
    for s, e in segments:
        if (e - s) >= min_segment_height:
            cropped_imgs.append(img[s:e, :])
            final_segments.append((s, e))

    return cropped_imgs, final_segments


img_path = "data/test_samples/sample03_processed.jpg"
img = io.imread(img_path, as_gray=True)

cropped_imgs, segments = split_gels_horizontal_safe(img, min_gap=55,
                                                    buffer=30,
                                                    min_segment_height=100)
print(f"Detected {len(cropped_imgs)} gel(s)")

# vizualizácia
fig, ax = plt.subplots(1, max(len(cropped_imgs), 1), figsize=(15, 5))
ax = np.atleast_1d(ax)
for i, g in enumerate(cropped_imgs):
    ax[i].imshow(g, cmap='gray')
    ax[i].axis('off')
plt.show()
