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

def get_signals_list(ing_height, ing_width):
    signals = []
    lowest_signal = 255

    for pixel_line in range(img_width):
        signal_line = grayscale_img[:,pixel_line]
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


                

################# MAIN CODE

# read the preprocessed image
image = sys.argv[1]
grayscale_img = cv2.imread(image, cv2.IMREAD_GRAYSCALE) # to fix: move to img preprocessing
img_height, img_width = grayscale_img.shape

signals = get_signals_list(img_height, img_width)

signal_simplified = get_simplified_pixel_line_signal(signals, 200, 17)

print(get_peak_specifs(signal_simplified, 10))

plot_signal_in_pixel_line(200)
show_image()

