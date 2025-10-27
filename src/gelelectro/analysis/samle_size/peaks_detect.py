import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy import signal as sig

################ DEFINE FUNCTIONS
def show_image():
    plt.figure()
    plt.imshow(grayscale_img, cmap = 'gray')
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

    if lowest_signal != 0:
        print("normalize the intensity scale")
    
    return signals

def plot_signal_in_pixel_line(pixel_line):
    plt.figure()
    plt.plot(signals[pixel_line])
    plt.title(pixel_line)
    plt.show(block = False)
    plt.ylim(top=255)

def get_peak_centers(pixel_line, width_threshold, intensity_threshold):
    signal = signals[pixel_line]
    peak_centers = []
    peaks, properties = sig.find_peaks(signal, height=intensity_threshold, width=width_threshold)

    for peak in peaks:
        peak_centers.append(peak)
    
    return peak_centers

def get_lane_and_ladder_specifs(width_threshold, min_ladder_bands, all_peak_centers):
    all_lane_specifs = {}
    ladder_specifs = {}
    ladder_signs = 0
    start = False
    width = 0

    for pixel_line, peak_center in enumerate(all_peak_centers): # to fix: no need to enumerate
        num_of_peaks = len(all_peak_centers[pixel_line])
        #print(num_of_peaks)

        if num_of_peaks > 0 and not (start or ladder_signs > 0):
            #print("start")
            if num_of_peaks > min_ladder_bands:
                #print("ladder start")
                ladder_signs += 1
            lane_specifs = {}
            start = pixel_line
            width = 1
                
            
        elif num_of_peaks > 0 and start:
            if num_of_peaks > min_ladder_bands:
                ladder_signs += 1
                #print("ladder")
            else:
                ladder_signs += -1
            width += 1 

        elif num_of_peaks == 0 and (start or pixel_line == len(all_lane_specifs)-1):
            #print("end")
            if width > width_threshold:
                #print("width satisfied")
                lane_specifs["start"] = start
                lane_specifs["center"] = (start + width/2)
                lane_specifs["end"] = pixel_line

                if ladder_signs > width*(2/3):
                    #print("ladder end")
                    ladder_specifs[len(ladder_specifs)] = lane_specifs
                else:
                    #print("lane end")
                    all_lane_specifs[len(all_lane_specifs)] = lane_specifs
            width = 0
            start = False
            ladder_signs = 0

    return all_lane_specifs, ladder_specifs

def get_max_signal_distance_of_lane(lane):
    center_pixel_line = int(all_lane_specifs[lane]["center"])
    center_signal = np.array(signals[center_pixel_line])
    distance = np.argmax(center_signal)
    return distance

def get_sample_size_function_variables(ladder_specifs, sizes):
    ladder_center_signal = signals[int(ladder_specifs[0]["center"])]
    distances, properties = sig.find_peaks(ladder_center_signal, height=110, distance = 15, prominence=5)

    log_sizes = np.log10(sizes)

    slope, intercept = np.polyfit(distances, log_sizes, 1)

    return slope, intercept

def estimate_sample_size(slope, intercept, distance):
    predicted_size = 10**(slope * distance + intercept)
    return predicted_size
    
################# MAIN CODE

# read the preprocessed image
image = sys.argv[1]
grayscale_img = cv2.imread(image, cv2.IMREAD_GRAYSCALE) # to fix: move to img preprocessing
img_height, img_width = grayscale_img.shape

signals = get_signals_list(grayscale_img, img_width)

all_peak_centers = {}

for index, signal in enumerate(signals):
    all_peak_centers[index] = get_peak_centers(index, 5, 40)

max_peaks_in_line = max([len(all_peak_centers[i]) for i in range(len(all_peak_centers))])

all_lane_specifs, ladder_specifs = get_lane_and_ladder_specifs(width_threshold = 25, min_ladder_bands = 3, all_peak_centers = all_peak_centers)

# get variables of the distance->size function
slope, intercept = get_sample_size_function_variables(ladder_specifs, [2000, 1500, 600])

#estimate the size of each sample
for lane in all_lane_specifs:
    sample_distance = get_max_signal_distance_of_lane(lane) # to fix: mby compute avg of the lane
    sample_size = estimate_sample_size(slope, intercept, sample_distance)
    print(sample_size)
#plot_signal_in_pixel_line(104)



show_image()