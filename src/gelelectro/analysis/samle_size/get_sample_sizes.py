import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy import signal as sig
from utils import *

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


# get variables of the distance->size function
slope, intercept, main_ladder_band_dist = get_sample_size_function_variables(signals, ladder_specifs, [1517, 1000, 517])

#estimate the size of each sample
for lane in all_lane_specifs:
    sample_distance = get_max_signal_distance_of_lane(signals, all_lane_specifs, lane) # to fix: mby compute avg of the lane
    sample_size = estimate_sample_size(slope, intercept, sample_distance)
    print(sample_size)

plot_signal_in_pixel_line(signals, int(ladder_specifs["center"]))

print(main_ladder_band_dist)

show_image(grayscale_img)
