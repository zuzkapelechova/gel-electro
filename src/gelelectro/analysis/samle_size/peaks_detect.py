import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys

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
    plt.plot(signals[pixel_line])
    plt.show(block = False)
    plt.ylim(top=255)


def get_simplified_pixel_line_signal(signals, pixel_line, intensity_threshold):
    signal = signals[pixel_line]
    signal_simplified = []

    for pixel in signal:
        if pixel > intensity_threshold:
            signal_simplified.append(True)
        else:
            signal_simplified.append(False)  

    return signal_simplified

def get_peak_specifs(signal_simplified, length_threshold):
    start = False
    length = 0
    peak_specifs = {
        "starts": [],
        "centers": [],
        "ends": []
    }

    for index, value in enumerate(signal_simplified):
        
        if value and not start:
            start = index
            length = 1
        
        elif value and start:
            length += 1
        
        elif not value and start:
            length = index - start
            if length > length_threshold:
                center = (index - (length/2))
                peak_specifs["starts"].append(start)
                peak_specifs["centers"].append(center)
                peak_specifs["ends"].append(index)
            length = 0
            start = False
    
    return peak_specifs


def get_lane_and_ladder_specifs(width_threshold, min_ladder_bands, all_peak_specifs):
    all_lane_specifs = {}
    ladder_specifs = {}
    ladder_start = False
    start = False
    width = 0

    for pixel_line, peak_specifs in enumerate(all_peak_specifs):
        if len(all_peak_specifs[peak_specifs]["centers"]) > 0 and not (start or ladder_start):
            if len(all_peak_specifs[peak_specifs]["centers"]) > min_ladder_bands:
                ladder_start = True
            lane_specifs = {}
            start = pixel_line
            width = 1
                
            
        elif len(all_peak_specifs[peak_specifs]["centers"]) and start:
            if len(all_peak_specifs[peak_specifs]["centers"]) > min_ladder_bands:
                ladder_start = True
            width += 1 

        elif len(all_peak_specifs[peak_specifs]["centers"]) == 0 and (start or pixel_line == len(all_lane_specifs)-1):
            if width > width_threshold:
                lane_specifs["start"] = start
                lane_specifs["center"] = (start + width/2)
                lane_specifs["end"] = pixel_line
                if ladder_start:
                    ladder_specifs[len(all_lane_specifs)] = lane_specifs
                else:
                    all_lane_specifs[len(all_lane_specifs)] = lane_specifs
            width = 0
            start = False
            ladder_start = False

    return all_lane_specifs, ladder_specifs

def get_max_signal_distance_of_lane(lane):
    center_pixel_line = int(lane_specifs[lane]["center"])
    center_signal = np.array(signals[center_pixel_line])
    distance = np.argmax(center_signal)
    return distance

################# MAIN CODE

# read the preprocessed image
image = sys.argv[1]
grayscale_img = cv2.imread(image, cv2.IMREAD_GRAYSCALE) # to fix: move to img preprocessing
img_height, img_width = grayscale_img.shape

signals = get_signals_list(grayscale_img, img_width)

all_peak_specifs = {}
for index, signal in enumerate(signals):
    signal_simplified = get_simplified_pixel_line_signal(signals, index, 17)
    all_peak_specifs[index] = get_peak_specifs(signal_simplified, 10)

max_peaks_in_line = max([len(all_peak_specifs[i]["centers"]) for i in range(len(all_peak_specifs))])

lane_specifs, ladder_specifs = get_lane_and_ladder_specifs(width_threshold = 7, min_ladder_bands = 4, all_peak_specifs = all_peak_specifs)

# estimate the size of each sample
for lane in lane_specifs:
    #print(get_max_signal_distance_of_lane(lane)) # mby compute avg of the lane
    print(lane_specifs[lane]["center"])





show_image()