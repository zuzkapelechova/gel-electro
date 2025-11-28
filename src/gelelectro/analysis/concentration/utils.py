import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy import signal as sig


def save_image(img, output):
    plt.figure()
    plt.imshow(img, cmap = 'gray')
    plt.axis('off')
    #plt.show()
    plt.savefig(output, dpi=300, bbox_inches='tight')

def get_signals_list(img, img_width):
    # returns list of signals (per pixel line)
    signals = []

    for pixel_line in range(img_width):
        signal_line = img[:,pixel_line]
        signals.append(signal_line)
        
    return signals

def plot_signal_in_pixel_line(signals, pixel_line):
    plt.figure()
    plt.plot(signals[pixel_line])
    plt.title(pixel_line)
    plt.ylim(top=255)
    plt.show()

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
    
def get_peak_edges(signals, all_peak_centers, pixel_line, abs_intens_threshold, ladder_peak_centers = None):

    signal = signals[pixel_line]

    if ladder_peak_centers is not None:
        peak_centers = ladder_peak_centers
    else:
        peak_centers = all_peak_centers[pixel_line]

    starts = {}
    ends = {}
    
    for i, peak_center in enumerate(peak_centers):            

        if len(starts) == 0 and len(peak_centers) == 1:    # first and single bend

            start_to_peak_signal = [signal[pixel] for pixel in range(0, peak_centers[i])]
            secondary_peak_centers_start, secondary_peak_properties_start = sig.find_peaks(start_to_peak_signal, height = 5)
            peak_to_end_signal = [signal[pixel] for pixel in range(peak_centers[i], len(signal))]
            secondary_peak_centers_end, secondary_peak_properties_end = sig.find_peaks(peak_to_end_signal, height = 5)
            
            inv_start_to_peak_signal = [255 - signal[pixel] for pixel in range(0, peak_centers[i])]
            inv_secondary_peak_centers_start, inv_secondary_peak_properties_start = sig.find_peaks(inv_start_to_peak_signal, height = 5)
            inv_peak_to_end_signal = [255 - signal[pixel] for pixel in range(peak_centers[i], len(signal))]
            inv_secondary_peak_centers_end, inv_secondary_peak_properties_end = sig.find_peaks(inv_peak_to_end_signal, height = 5)

            # start edge
            if len(secondary_peak_centers_start) >= 1:
                print("IF", secondary_peak_centers_start)
                peak_to_sec_peak_dist_start = len(start_to_peak_signal) - max(secondary_peak_centers_start)
            else:
                print("ELSE", secondary_peak_centers_start)
                peak_to_sec_peak_dist_start = 99999999999999

            if len(inv_secondary_peak_centers_start) >= 1:
                inv_peak_to_sec_peak_dist_start = len(inv_start_to_peak_signal) - max(inv_secondary_peak_centers_start)
            else:
                print("ELSE", secondary_peak_centers_start)
                inv_peak_to_sec_peak_dist_start = 99999999999999

            print(peak_center, min(peak_to_sec_peak_dist_start, inv_peak_to_sec_peak_dist_start))

            starts[peak_center] = peak_center - min(peak_to_sec_peak_dist_start, inv_peak_to_sec_peak_dist_start)

            # end edge    
            if len(secondary_peak_centers_end) >= 1:
                peak_to_sec_peak_dist_end = min(secondary_peak_centers_end)
            else:
                peak_to_sec_peak_dist_end = 999999999999999

            if len(inv_secondary_peak_centers_end) >= 1:
                inv_peak_to_sec_peak_dist_end = min(inv_secondary_peak_centers_end)
            else:
                inv_peak_to_sec_peak_dist_end = 999999999999999
            print("END", peak_center, min(peak_to_sec_peak_dist_end, inv_peak_to_sec_peak_dist_end))
            ends[peak_center] = peak_center + min(peak_to_sec_peak_dist_end, inv_peak_to_sec_peak_dist_end)

        elif len(starts) == 0:    # first bend

            start_to_peak_signal = [signal[pixel] for pixel in range(0, peak_centers[i])]
            secondary_peak_centers_start, secondary_peak_properties_start = sig.find_peaks(start_to_peak_signal, height = 5)
            peak_to_peak_signal_end = [signal[pixel] for pixel in range(peak_centers[i], peak_centers[i+1])]
            secondary_peak_centers_end, secondary_peak_properties_end = sig.find_peaks(peak_to_peak_signal_end, height = 5)
            
            inv_start_to_peak_signal = [255 - signal[pixel] for pixel in range(0, peak_centers[i])]
            inv_secondary_peak_centers_start, inv_secondary_peak_properties_start = sig.find_peaks(inv_start_to_peak_signal, height = 5)
            inv_peak_to_peak_signal_end = [255 - signal[pixel] for pixel in range(peak_centers[i], peak_centers[i+1])]
            inv_secondary_peak_centers_end, inv_secondary_peak_properties_end = sig.find_peaks(inv_peak_to_peak_signal_end, height = 5)

            # start edge
            if len(secondary_peak_centers_start) >= 1:
                print("IF", secondary_peak_centers_start)
                peak_to_sec_peak_dist_start = len(start_to_peak_signal) - max(secondary_peak_centers_start)
            else:
                print("ELSE", secondary_peak_centers_start)
                peak_to_sec_peak_dist_start = 99999999999999

            if len(inv_secondary_peak_centers_start) >= 1:
                inv_peak_to_sec_peak_dist_start = len(inv_start_to_peak_signal) - max(inv_secondary_peak_centers_start)
            else:
                print("ELSE", secondary_peak_centers_start)
                inv_peak_to_sec_peak_dist_start = 99999999999999

            print(peak_center, min(peak_to_sec_peak_dist_start, inv_peak_to_sec_peak_dist_start))

            starts[peak_center] = peak_center - min(peak_to_sec_peak_dist_start, inv_peak_to_sec_peak_dist_start)

            # end edge    
            if len(secondary_peak_centers_end) >= 1:
                peak_to_sec_peak_dist_end = min(secondary_peak_centers_end)
            else:
                peak_to_sec_peak_dist_end = 999999999999999

            if len(inv_secondary_peak_centers_end) >= 1:
                inv_peak_to_sec_peak_dist_end = min(inv_secondary_peak_centers_end)
            else:
                inv_peak_to_sec_peak_dist_end = 999999999999999
            print("END", peak_center, min(peak_to_sec_peak_dist_end, inv_peak_to_sec_peak_dist_end))
            ends[peak_center] = peak_center + min(peak_to_sec_peak_dist_end, inv_peak_to_sec_peak_dist_end)

        
        elif len(starts) < (len(peak_centers) - 1):    # middle bands

            peak_to_peak_signal_start = [signal[pixel] for pixel in range(peak_centers[i-1], peak_centers[i])]
            secondary_peak_centers_start, secondary_peak_properties_start = sig.find_peaks(peak_to_peak_signal_start, height = 5)
            peak_to_peak_signal_end = [signal[pixel] for pixel in range(peak_centers[i-1], peak_centers[i])]
            secondary_peak_centers_end, secondary_peak_properties_end = sig.find_peaks(peak_to_peak_signal_end, height = 5)
            
            inv_peak_to_peak_signal_start = [255 - signal[pixel] for pixel in range(peak_centers[i-1], peak_centers[i])]
            inv_secondary_peak_centers_start, inv_secondary_peak_properties_start = sig.find_peaks(inv_peak_to_peak_signal_start, height = 5)
            inv_peak_to_peak_signal_end = [255 - signal[pixel] for pixel in range(peak_centers[i-1], peak_centers[i])]
            inv_secondary_peak_centers_end, inv_secondary_peak_properties_end = sig.find_peaks(inv_peak_to_peak_signal_end, height = 5)

            # start edge
            if len(secondary_peak_centers_start) >= 1:
                peak_to_sec_peak_dist_start = len(peak_to_peak_signal_start) - max(secondary_peak_centers_start)
            else:
                peak_to_sec_peak_dist_start = 99999999999999

            if len(inv_secondary_peak_centers_start) >= 1:
                inv_peak_to_sec_peak_dist_start = len(inv_peak_to_peak_signal_start) - max(inv_secondary_peak_centers_start)
            else:
                inv_peak_to_sec_peak_dist_start = 99999999999999
            print(peak_center, min(peak_to_sec_peak_dist_start, inv_peak_to_sec_peak_dist_start))
            starts[peak_center] = peak_center - min(peak_to_sec_peak_dist_start, inv_peak_to_sec_peak_dist_start)

            # end edge    
            if len(secondary_peak_centers_end) >= 1:
                peak_to_sec_peak_dist_end = min(secondary_peak_centers_end)
            else:
                peak_to_sec_peak_dist_end = 999999999999999

            if len(inv_secondary_peak_centers_end) >= 1:
                inv_peak_to_sec_peak_dist_end = min(inv_secondary_peak_centers_end)
            else:
                inv_peak_to_sec_peak_dist_end = 999999999999999
            print("END", peak_center, min(peak_to_sec_peak_dist_end, inv_peak_to_sec_peak_dist_end))
            ends[peak_center] = peak_center + min(peak_to_sec_peak_dist_end, inv_peak_to_sec_peak_dist_end)

        else:   # last band

            peak_to_peak_signal_start = [signal[pixel] for pixel in range(peak_centers[i-1], peak_centers[i])]
            secondary_peak_centers_start, secondary_peak_properties_start = sig.find_peaks(peak_to_peak_signal_start, height = 5)
            peak_to_end_signal = [signal[pixel] for pixel in range(peak_centers[i], len(signal))]
            secondary_peak_centers_end, secondary_peak_properties_end = sig.find_peaks(peak_to_end_signal, height = 5)
            
            inv_peak_to_peak_signal_start = [255 - signal[pixel] for pixel in range(peak_centers[i-1], peak_centers[i])]
            inv_secondary_peak_centers_start, inv_secondary_peak_properties_start = sig.find_peaks(inv_peak_to_peak_signal_start, height = 5)
            inv_peak_to_end_signal = [255 - signal[pixel] for pixel in range(peak_centers[i], len(signal))]
            inv_secondary_peak_centers_end, inv_secondary_peak_properties_end = sig.find_peaks(inv_peak_to_end_signal, height = 5)

            # start edge
            if len(secondary_peak_centers_start) >= 1:
                peak_to_sec_peak_dist_start = len(peak_to_peak_signal_start) - max(secondary_peak_centers_start)
            else:
                peak_to_sec_peak_dist_start = 99999999999999

            if len(inv_secondary_peak_centers_start) >= 1:
                inv_peak_to_sec_peak_dist_start = len(inv_peak_to_peak_signal_start) - max(inv_secondary_peak_centers_start)
            else:
                inv_peak_to_sec_peak_dist_start = 99999999999999
            print(peak_center, min(peak_to_sec_peak_dist_start, inv_peak_to_sec_peak_dist_start))
            starts[peak_center] = peak_center - min(peak_to_sec_peak_dist_start, inv_peak_to_sec_peak_dist_start)

            # end edge    
            if len(secondary_peak_centers_end) >= 1:
                peak_to_sec_peak_dist_end = min(secondary_peak_centers_end)
            else:
                peak_to_sec_peak_dist_end = 999999999999999

            if len(inv_secondary_peak_centers_end) >= 1:
                inv_peak_to_sec_peak_dist_end = min(inv_secondary_peak_centers_end)
            else:
                inv_peak_to_sec_peak_dist_end = 999999999999999
            print("END", peak_center, min(peak_to_sec_peak_dist_end, inv_peak_to_sec_peak_dist_end))
            ends[peak_center] = peak_center + min(peak_to_sec_peak_dist_end, inv_peak_to_sec_peak_dist_end)


    return starts, ends

def get_intensities(all_peak_centers, signals, pixel_line, ladder_peak_centers = None):
    if ladder_peak_centers is not None:
        starts, ends = get_peak_edges(signals, all_peak_centers, pixel_line, 10, ladder_peak_centers)
    else:
        starts, ends = get_peak_edges(signals, all_peak_centers, pixel_line, 10)
    
    intensities = {}
    for peak_center in starts.keys():
        signal = signals[pixel_line]
        intensity = sum(int(signal[pixel]) for pixel in range(starts[peak_center], ends[peak_center]))
        intensities[peak_center] = int(intensity)
    return intensities

def mk_weight_fnc(ladder_specifs, all_peak_centers, signals, weights, plot_output):

    ladder_center_signal = signals[int(ladder_specifs["center"])]
    distances, properties = sig.find_peaks(ladder_center_signal, height=5, distance = 5, prominence=5)


    if len(distances) > 3:  # doubled - mk fnc()
    # Get heights of detected peaks # doubled - mk fnc()
        peak_heights = properties['peak_heights']   # doubled - mk fnc()
        top_indices = np.argsort(peak_heights)[-3:] # doubled - mk fnc()
    # doubled - mk fnc()
        # Select the highest 3 peaks    # doubled - mk fnc()
        top_peak_distances = sorted(distances[top_indices]) # doubled - mk fnc()
    elif len(distances) == 3:   # doubled - mk fnc()
        top_peak_distances = sorted(distances)  # If less than 3 peaks, take whatever is found  # doubled - mk fnc()
    else:   # doubled - mk fnc()
        raise ValueError("Couldn't find 3 main ladder bands")   # doubled - mk fnc()

    intensities = get_intensities(all_peak_centers, signals, int(ladder_specifs["center"]), top_peak_distances)
    
    areas = [intensities[top_peak_distances[0]], intensities[top_peak_distances[1]], intensities[top_peak_distances[2]]]

    a, b = np.polyfit(areas, weights, 1)

    x_line = np.linspace(areas[0], areas[2])
    y_line = a * x_line + b

    plt.plot(x_line, y_line, color="red")
    plt.scatter(areas, weights)
    plt.xlabel("area under peak")
    plt.ylabel("weight [ng]")
    plt.title("area under peak - weight\nstandard curve")
    plt.show()

    return a, b

def get_sample_weight(all_peak_centers, signals, pixel_line, a, b):
    intensities = get_intensities(all_peak_centers, signals, pixel_line)
    weights = {}
    for peak_center in intensities.keys():
        weight_estimation = a * intensities[peak_center] + b
        if weight_estimation < 1:
            weight = "< 1"
        else:
            weight = weight_estimation
        weights[peak_center] = weight
    return weights

def get_band_distances_of_lane(signals, all_lane_specifs, lane):
    center_pixel_line = int(all_lane_specifs[lane]["center"])
    center_band_distances = np.array(signals[center_pixel_line])
    return center_band_distances

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
        raise ValueError("Couldn't find 3 main ladder bands")


    log_sizes = np.log10(sizes)

    slope, intercept = np.polyfit(top_peak_distances, log_sizes, 1)

    return slope, intercept, top_peak_distances

def save_sample_size_function_plot(slope, intercept, top_peak_distances, sizes, output):
    
    x = np.linspace(min(top_peak_distances), max(top_peak_distances), 100)

    # Compute corresponding y-values using the linear function
    y = slope * x + intercept

    # Plot the data points and the fitted line
    plt.figure()
    plt.scatter(top_peak_distances, np.log10(sizes))
    plt.plot(x, y, color='red')
    plt.xlabel('Distance [px]')
    plt.ylabel('log10(size)')
    plt.title('SIZE - DISTANCE')
    #plt.show()
    plt.savefig(output)

def estimate_sample_size(slope, intercept, distance):
    predicted_size = 10**(slope * distance + intercept)
    return predicted_size
