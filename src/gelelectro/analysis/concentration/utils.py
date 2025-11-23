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
        
        top_intens = signal[peak_center]
        start_pixel = end_pixel = peak_center
        end_found = False
        start_found = False
        while not start_found or not end_found:
            end_pixel += 1
            start_pixel -= 1
            
            
            if len(starts) == 0:    # first bend

                start_to_peak_signal = [signal[pixel] for pixel in range(0, peak_centers[i])]
                secondary_peak_centers_start, secondary_peak_properties_start = sig.find_peaks(start_to_peak_signal, height = 5)
                peak_to_peak_signal_end = [signal[pixel] for pixel in range(peak_centers[i], peak_centers[i+1])]
                secondary_peak_centers_end, secondary_peak_properties_end = sig.find_peaks(peak_to_peak_signal_end, height = 5)

                if len(peak_to_peak_signal_end) >= 1 or len(start_to_peak_signal) >= 1:
                    minimum_dist_end = abs(np.argmin(peak_to_peak_signal_end))
                    minimum_dist_start = abs(peak_center - np.argmin(start_to_peak_signal))
                    peak_to_loc_min_dist = min(minimum_dist_end, minimum_dist_start)
                else:
                    peak_to_loc_min_dist = 99999999999
                if len(secondary_peak_centers_start) >= 1:
                    peak_to_sec_peak_dist_start = max(secondary_peak_centers_start) - peak_center
                else:
                    peak_to_sec_peak_dist_start = 99999999999999
                if len(secondary_peak_centers_end) >= 1:
                    peak_to_sec_peak_dist_end = min(secondary_peak_centers_end)
                else:
                    peak_to_sec_peak_dist_end = 999999999999999

                min_abs_ndx = np.argmin([abs(peak_to_loc_min_dist), abs(peak_to_sec_peak_dist_end), abs(peak_to_sec_peak_dist_start)])
                threshold_value_distance = [peak_to_loc_min_dist, peak_to_sec_peak_dist_end, peak_to_sec_peak_dist_start][min_abs_ndx]
                threshold = signal[peak_center + threshold_value_distance]

                if signal[end_pixel] <= threshold and not end_found:
                    ends[peak_center] = end_pixel
                    end_found = True
                if signal[start_pixel] <= threshold and not start_found: # mby abs thrshld
                    starts[peak_center] = start_pixel
                    start_found = True
            
            elif len(starts) < (len(peak_centers) - 1):    # middle bands

                peak_to_peak_signal_start = [signal[pixel] for pixel in range(peak_centers[i-1], peak_centers[i])]
                secondary_peak_centers_start, secondary_peak_properties_start = sig.find_peaks(peak_to_peak_signal_start, height = 5)
                peak_to_peak_signal_end = [signal[pixel] for pixel in range(peak_centers[i-1], peak_centers[i])]
                secondary_peak_centers_end, secondary_peak_properties_end = sig.find_peaks(peak_to_peak_signal_end, height = 5)

                if len(peak_to_peak_signal_end) >= 1 or len(peak_to_peak_signal_start) >= 1:
                    minimum_dist_end = abs(np.argmin(peak_to_peak_signal_end))
                    minimum_dist_start = abs(len(peak_to_peak_signal_start) - np.argmin(peak_to_peak_signal_start))
                    peak_to_loc_min_dist = min(minimum_dist_end, minimum_dist_start)
                else:
                    peak_to_loc_min_dist = 99999999999
                if len(secondary_peak_centers_start) >= 1:
                    peak_to_sec_peak_dist_start = max(secondary_peak_centers_start) - peak_center
                else:
                    peak_to_sec_peak_dist_start = 99999999999999
                if len(secondary_peak_centers_end) >= 1:
                    peak_to_sec_peak_dist_end = min(secondary_peak_centers_end)
                else:
                    peak_to_sec_peak_dist_end = 999999999999999

                min_abs_ndx = np.argmin([abs(peak_to_loc_min_dist), abs(peak_to_sec_peak_dist_end), abs(peak_to_sec_peak_dist_start)])
                threshold_value_distance = [peak_to_loc_min_dist, peak_to_sec_peak_dist_end, peak_to_sec_peak_dist_start][min_abs_ndx]
                threshold = signal[peak_center + threshold_value_distance]

                if signal[end_pixel] <= threshold and not end_found:
                    ends[peak_center] = end_pixel
                    end_found = True
                if signal[start_pixel] <= threshold and not start_found: # mby abs thrshld
                    starts[peak_center] = start_pixel
                    start_found = True

            else:   # last band

                peak_to_peak_signal_start = [signal[pixel] for pixel in range(peak_centers[i-1], peak_centers[i])]
                secondary_peak_centers_start, secondary_peak_properties_start = sig.find_peaks(peak_to_peak_signal_start, height = 5)
                peak_to_end_signal = [signal[pixel] for pixel in range(peak_centers[i], len(signal))]
                secondary_peak_centers_end, secondary_peak_properties_end = sig.find_peaks(peak_to_end_signal, height = 5)
                
                if len(peak_to_end_signal) >= 1 or len(peak_to_peak_signal_start) >= 1:
                    minimum_dist_end = abs(np.argmin(peak_to_peak_signal_end))
                    minimum_dist_start = abs(len(peak_to_peak_signal_start) - np.argmin(peak_to_peak_signal_start))
                    peak_to_loc_min_dist = min(minimum_dist_end, minimum_dist_start)
                else:
                    peak_to_loc_min_dist = 99999999999
                if len(secondary_peak_centers_start) >= 1:
                    peak_to_sec_peak_dist_start = max(secondary_peak_centers_start) - peak_center
                else:
                    peak_to_sec_peak_dist_start = 99999999999999
                if len(secondary_peak_centers_end) >= 1:
                    peak_to_sec_peak_dist_end = min(secondary_peak_centers_end)
                else:
                    peak_to_sec_peak_dist_end = 999999999999999

                min_abs_ndx = np.argmin([abs(peak_to_loc_min_dist), abs(peak_to_sec_peak_dist_end), abs(peak_to_sec_peak_dist_start)])
                threshold_value_distance = [peak_to_loc_min_dist, peak_to_sec_peak_dist_end, peak_to_sec_peak_dist_start][min_abs_ndx]
                threshold = signal[peak_center + threshold_value_distance]

                if signal[end_pixel] <= threshold and not end_found:
                    ends[peak_center] = end_pixel
                    end_found = True
                if signal[start_pixel] <= threshold and not start_found: # mby abs thrshld
                    starts[peak_center] = start_pixel
                    start_found = True
    return starts, ends


'''
def get_avg_intens(all_lane_specifs, lane, signals, all_peak_centers):
    for pixel_line in range(all_lane_specifs[lane]["start"], all_lane_specifs[lane]["end"]):
        
        starts, ends = get_peak_edges(all_lane_specifs, signals, all_peak_centers, pixel_line, 5)
        sums = {}

        for peak_center in starts.keys():
            sum[peak_center]
'''
def get_intensities(all_peak_centers, signals, pixel_line, ladder_peak_centers = None):
    if ladder_peak_centers is not None:
        starts, ends = get_peak_edges(signals, all_peak_centers, pixel_line, 10, ladder_peak_centers)
        print(starts, ends)
    else:
        starts, ends = get_peak_edges(signals, all_peak_centers, pixel_line, 10)
    intensities = {}
    for peak_center in starts.keys():
        signal = signals[pixel_line]
        intensity = sum(int(signal[pixel]) for pixel in range(starts[peak_center], ends[peak_center]))
        intensities[peak_center] = int(intensity)
    return intensities

def mk_weight_fnc(ladder_specifs, all_peak_centers, signals, weights):

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
    
    values = np.log([intensities[top_peak_distances[0]], intensities[top_peak_distances[1]], intensities[top_peak_distances[2]]])

    plt.scatter(values, weights)
    plt.show()

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
