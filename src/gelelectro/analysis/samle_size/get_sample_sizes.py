import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy import signal as sig
from utils import *

image = sys.argv[1] # preprocessed image
ladder2 = sys.argv[2] # ladder on each side - True/False

# read the preprocessed image
png_image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
cv2.imwrite('image.jpg', png_image) # to do: dlt this line, move to img preprocessing
image = 'image.jpg'
grayscale_img = cv2.imread(image, cv2.IMREAD_GRAYSCALE) # to fix: move to img preprocessing
img_height, img_width = grayscale_img.shape

# read the img into signals dictionary
signals = get_signals_list(grayscale_img, img_width)

# get positions of signal peaks per pixel line
all_peak_centers = {}
for index, signal in enumerate(signals):
    all_peak_centers[index] = get_peak_centers(signals, index, 5, 20)

# find ladder and lanes and save the specifications (start, center, end of each lane)
all_lane_specifs, ladder_specifs = get_lane_and_ladder_specifs(width_threshold = 25, min_ladder_bands = 3, all_peak_centers = all_peak_centers)

# get variables of the distance->size function
slope, intercept, main_ladder_band_dist = get_sample_size_function_variables(signals, ladder_specifs, [1517, 1000, 517])



color_img = cv2.cvtColor(grayscale_img, cv2.COLOR_GRAY2RGB)

# variable for cv2.putText()
text_size = cv2.getTextSize("-----", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

results_text = ""

# estimate the size of each sample
for lane in all_lane_specifs:
    sample_distances = all_peak_centers[int(all_lane_specifs[lane]["center"])]
    sample_sizes = []

    results_text += f"lane {lane}: \n"
    for i, sample_distance in enumerate(sample_distances):
        sample_size = estimate_sample_size(slope, intercept, sample_distance)
        sample_sizes.append(sample_size)

        results_text += f"{str(sample_size)} bp\n"

        # add image annotations
        centered_x_position = int(all_lane_specifs[lane]["center"]) - text_size[0] // 2
        y_position = int(sample_distance)
        cv2.putText(color_img, f"{int(sample_size)} bp", (centered_x_position, 15 + 15*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
        cv2.putText(color_img, "-----", (centered_x_position, sample_distance), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)

    results_text += "--------------\n"

centered_x_ladder_position = int(ladder_specifs["center"] - text_size[0] // 2)
sizes = [1517, 1000, 517]
for i, y_band_position in enumerate(main_ladder_band_dist):
    cv2.putText(color_img, f"{sizes[i]} bp", (centered_x_ladder_position, 15 + 15*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
    cv2.putText(color_img, "-----", (centered_x_ladder_position, y_band_position), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)



return grayscale_img, results_text

