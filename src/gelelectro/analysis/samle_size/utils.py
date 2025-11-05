import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy import signal as sig


def show_image(img):
    plt.figure()
    plt.imshow(img, cmap = 'gray')
    plt.axis('off')
    plt.show()

def get_signals_list(img, img_width):
    signals = []
    lowest_signal = 255

    for pixel_line in range(img_width):
        signal_line = img[:,pixel_line]
        signals.append(signal_line)
        
        if min(signal_line) < lowest_signal: # to fix: mby already in preprocessing
            lowest_signal = min(signal_line)
    
    return signals

def plot_signal_in_pixel_line(signals, pixel_line):
    plt.figure()
    plt.plot(signals[pixel_line])
    plt.title(pixel_line)
    plt.show(block = False)
    plt.ylim(top=255)

def get_peak_centers(signals, pixel_line, width_threshold, intensity_threshold):
    signal = signals[pixel_line]
    peak_centers = []
    peaks, properties = sig.find_peaks(signal, height=intensity_threshold, width=width_threshold)

    for peak in peaks:
        peak_centers.append(peak)
    
    return peak_centers

def get_lane_and_ladder_specifs(width_threshold, min_ladder_bands, all_peak_centers):
    all_lane_specifs = {}
    ladder_specifs = []
    start = False
    width = 0

    for pixel_line in all_peak_centers:
        num_of_peaks = len(all_peak_centers[pixel_line])

        if num_of_peaks > 0 and not start:
            lane_specifs = {}
            start = pixel_line
            width = 1
                  
        elif num_of_peaks > 0 and start:
            width += 1 

        elif num_of_peaks == 0 and (start or pixel_line == len(all_lane_specifs)-1):
            if width > width_threshold:
                lane_specifs["start"] = start
                lane_specifs["center"] = (start + width/2)
                lane_specifs["end"] = pixel_line

                if len(ladder_specifs) == 0:
                    ladder_specifs = lane_specifs
                else:
                    all_lane_specifs[len(all_lane_specifs)] = lane_specifs

            width = 0
            start = False

    return all_lane_specifs, ladder_specifs

def get_max_signal_distance_of_lane(signals, all_lane_specifs, lane):
    center_pixel_line = int(all_lane_specifs[lane]["center"])
    center_signal = np.array(signals[center_pixel_line])
    distance = np.argmax(center_signal)
    return distance

def get_sample_size_function_variables(signals, ladder_specifs, sizes):
    ladder_center_signal = signals[int(ladder_specifs["center"])]
    distances, properties = sig.find_peaks(ladder_center_signal, height=5, distance = 5, prominence=5)

    if len(distances) > 3:
        # Get heights of detected peaks
        peak_heights = properties['peak_heights']
        top_indices = np.argsort(peak_heights)[-3:]

        # Select the highest 3 peaks
        top_peak_distances = sorted(distances[top_indices])
    elif len(distances) == 3:
        top_peak_distances = sorted(distances)  # If less than 3 peaks, take whatever is found
    else:
        raise ValueError("Couldn't find 3 main ladder bends")


    log_sizes = np.log10(sizes)

    slope, intercept = np.polyfit(top_peak_distances, log_sizes, 1)

    return slope, intercept, top_peak_distances

def plot_sample_size_function(slope, intercept, top_peak_distances, sizes):
    
    x = np.linspace(min(top_peak_distances), max(top_peak_distances), 100)

    # Compute corresponding y-values using the linear function
    y = slope * x + intercept

    # Plot the data points and the fitted line
    plt.figure()
    plt.scatter(top_peak_distances, np.log10(sizes))
    plt.plot(x, y, color='red')
    plt.xlabel('Distance [px]')
    plt.ylabel('log10(size)')
    plt.title('Distance - size')
    plt.show()

def estimate_sample_size(slope, intercept, distance):
    predicted_size = 10**(slope * distance + intercept)
    return predicted_size
