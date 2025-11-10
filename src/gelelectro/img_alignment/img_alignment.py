import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy import signal as sig
from utils import *

# use only if there are two ladders (one on each side)

# read the preprocessed image
image = sys.argv[1]
png_image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
cv2.imwrite('image.jpg', png_image) # to do: dlt this line, move to img preprocessing
image = 'image.jpg'
grayscale_img = cv2.imread(image, cv2.IMREAD_GRAYSCALE) # to fix: move to img preprocessing
img_height, img_width = grayscale_img.shape

signals = get_signals_list(grayscale_img, img_width)

all_peak_centers = {}

for index, signal in enumerate(signals):
    all_peak_centers[index] = get_peak_centers(signals, index, 5, 40)


max_peaks_in_line = max([len(all_peak_centers[i]) for i in range(len(all_peak_centers))])

all_lane_specifs, ladder_specifs = get_lane_and_ladder_specifs(width_threshold = 25, min_ladder_bands = 3, all_peak_centers = all_peak_centers)

# align the img based on the highest signal of each ladder
point1_x = ladder_specifs["center"]
point2_x = all_lane_specifs[max(all_lane_specifs.keys())]["center"]

point1_y = np.argmax(signals[int(point1_x)])
point2_y = np.argmax(signals[int(point2_x)])

delta_x = point1_x - point2_x
delta_y = point1_y - point2_y
angle = np.arctan2(delta_y, delta_x) * 180 / np.pi

center = [img_height/2, img_width/2]

M = cv2.getRotationMatrix2D(center, angle, 1.0)
aligned =cv2.warpAffine(grayscale_img, M, [img_width, img_height])

cv2.imshow('aligned', aligned)
cv2.waitKey(0)
cv2.destroyAllWindows()



