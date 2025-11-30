import numpy as np
from skimage import morphology, exposure
from skimage.util import img_as_float
import cv2
import matplotlib.pyplot as plt
from scipy import signal as sig
import base64
from io import BytesIO
from jinja2 import Template


class GelPreprocessor:
    """ Trieda na standardizaciu obrazkov """
    # preklapa obrazky aby boli vsetky biele na ciernom
    @staticmethod
    def detect_and_invert(img, low_perc=5, high_perc=95):
        img = img_as_float(img)
        p_low, p_high = np.percentile(img, [low_perc, high_perc])

        inverted = False
        if p_high > 0.5:
            img = 1 - img
            inverted = True

        params = {
            "detect_and_invert.low_percentile": low_perc,
            "detect_and_invert.high_percentile": high_perc,
            "detect_and_invert.p_low": float(p_low),
            "detect_and_invert.p_high": float(p_high),
            "detect_and_invert.inverted": inverted,
        }

        return img, params

    # rolling ball metoda (IOCBIO): adaptivne odstranovanie pozadia
    @staticmethod
    def adaptive_bg_subtraction(image: np.ndarray):
        mean_intensity = float(np.mean(image))
        std_intensity = float(np.std(image))
        contrast = std_intensity / (mean_intensity + 1e-5)

        radius = int(np.clip(200 * np.exp(-4 * contrast), 20, 180))

        selem = morphology.disk(radius)
        background = morphology.opening(image, selem)
        image_sub = image - background

        image_sub = exposure.rescale_intensity(image_sub, in_range="image", out_range=(0, 1))

        params = {
            "adaptive_bg_subtraction.mean_intensity": mean_intensity,
            "adaptive_bg_subtraction.std_intensity": std_intensity,
            "adaptive_bg_subtraction.contrast": float(contrast),
            "adaptive_bg_subtraction.radius": radius,
        }

        return image_sub, params

    # main function
    def process_image(self, img: np.ndarray):
        all_params = {}

        img_inverted, p1 = self.detect_and_invert(img)
        all_params.update(p1)

        img_processed, p2 = self.adaptive_bg_subtraction(img_inverted)
        all_params.update(p2)

        return img_processed, self.format_params(all_params)
    
    def format_params(self, params: dict) -> str:
        return "\n".join(f"{k}: {v}" for k, v in params.items())
        return image_sub


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

def get_bend_distances_of_lane(signals, all_lane_specifs, lane):
    center_pixel_line = int(all_lane_specifs[lane]["center"])
    center_bend_distances = np.array(signals[center_pixel_line])
    return center_bend_distances

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


def compute_sample_sizes(preprocessed_img, ladder2=False):
    grayscale_img = (preprocessed_img * 255).astype(np.uint8) if np.issubdtype(preprocessed_img.dtype, np.floating) else preprocessed_img.copy()
    
    img_height, img_width = grayscale_img.shape

    # read the img into signals dictionary
    signals = get_signals_list(grayscale_img, img_width)

    # get positions of signal peaks per pixel line
    all_peak_centers = {}
    for index, signal in enumerate(signals):
        all_peak_centers[index] = get_peak_centers(signals, index, 5, 20)

    # find ladder and lanes and save the specifications (start, center, end of each lane)
    all_lane_specifs, ladder_specifs = get_lane_and_ladder_specifs(width_threshold = 25, min_ladder_bands = 3, all_peak_centers = all_peak_centers)

    if ladder2 and len(all_lane_specifs) > 0:
        all_lane_specifs.popitem()

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
            sample_size = int(estimate_sample_size(slope, intercept, sample_distance))
            sample_sizes.append(sample_size)

            results_text += f"{str(sample_size)} bp\n"

            # add image annotations
            centered_x_position = int(all_lane_specifs[lane]["center"]) - text_size[0] // 2
            y_position = int(sample_distance)
            cv2.putText(color_img, f"{int(sample_size)} bp", (centered_x_position, 15 + 15*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
            cv2.putText(color_img, "-----", (centered_x_position, sample_distance), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 3, cv2.LINE_AA)

        results_text += "--------------\n"

    centered_x_ladder_position = int(ladder_specifs["center"] - text_size[0] // 2)
    sizes = [1517, 1000, 517]
    for i, y_band_position in enumerate(main_ladder_band_dist):
        cv2.putText(color_img, f"{sizes[i]} bp", (centered_x_ladder_position, 15 + 15*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
        cv2.putText(color_img, "-----", (centered_x_ladder_position, y_band_position), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2, cv2.LINE_AA)



    return color_img, results_text

def align_img(preprocessed_img):

    grayscale_img = (preprocessed_img * 255).astype(np.uint8) if np.issubdtype(preprocessed_img.dtype, np.floating) else preprocessed_img.copy()
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

    delta_x = point2_x - point1_x
    delta_y = point2_y - point1_y
    angle = np.arctan(delta_y / delta_x) * 180/np.pi

    mid_point = (int(point1_x + delta_x/2), int(point1_y + delta_y/2))


    M = cv2.getRotationMatrix2D(center=mid_point, angle=angle, scale=1)
    aligned =cv2.warpAffine(grayscale_img, M, dsize=(img_width, img_height))


    return aligned

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
                peak_to_sec_peak_dist_start = len(start_to_peak_signal) - max(secondary_peak_centers_start)
            else:
                peak_to_sec_peak_dist_start = 99999999999999

            if len(inv_secondary_peak_centers_start) >= 1:
                inv_peak_to_sec_peak_dist_start = len(inv_start_to_peak_signal) - max(inv_secondary_peak_centers_start)
            else:
                inv_peak_to_sec_peak_dist_start = 99999999999999

            if peak_to_sec_peak_dist_start > len(signal) and inv_peak_to_sec_peak_dist_start > len(signal):
                abs_min = np.argmin(start_to_peak_signal)
            else:
                abs_min = 9999999999999

            starts[peak_center] = peak_center - min(peak_to_sec_peak_dist_start, inv_peak_to_sec_peak_dist_start, abs_min)

            # end edge    
            if len(secondary_peak_centers_end) >= 1:
                peak_to_sec_peak_dist_end = min(secondary_peak_centers_end)
            else:
                peak_to_sec_peak_dist_end = 999999999999999

            if len(inv_secondary_peak_centers_end) >= 1:
                inv_peak_to_sec_peak_dist_end = min(inv_secondary_peak_centers_end)
            else:
                inv_peak_to_sec_peak_dist_end = 999999999999999

            if peak_to_sec_peak_dist_end > len(signal) and inv_peak_to_sec_peak_dist_end > len(signal):
                abs_min = np.argmin(peak_to_end_signal)
            else:
                abs_min = 9999999999999

            ends[peak_center] = peak_center + min(peak_to_sec_peak_dist_end, inv_peak_to_sec_peak_dist_end, abs_min)

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
                peak_to_sec_peak_dist_start = len(start_to_peak_signal) - max(secondary_peak_centers_start)
            else:
                peak_to_sec_peak_dist_start = 99999999999999

            if len(inv_secondary_peak_centers_start) >= 1:
                inv_peak_to_sec_peak_dist_start = len(inv_start_to_peak_signal) - max(inv_secondary_peak_centers_start)
            else:
                inv_peak_to_sec_peak_dist_start = 99999999999999

            if peak_to_sec_peak_dist_start > len(signal) and inv_peak_to_sec_peak_dist_start > len(signal):
                abs_min = np.argmin(start_to_peak_signal)
            else:
                abs_min = 9999999999999

            starts[peak_center] = peak_center - min(peak_to_sec_peak_dist_start, inv_peak_to_sec_peak_dist_start ,abs_min)

            # end edge    
            if len(secondary_peak_centers_end) >= 1:
                peak_to_sec_peak_dist_end = min(secondary_peak_centers_end)
            else:
                peak_to_sec_peak_dist_end = 999999999999999

            if len(inv_secondary_peak_centers_end) >= 1:
                inv_peak_to_sec_peak_dist_end = min(inv_secondary_peak_centers_end)
            else:
                inv_peak_to_sec_peak_dist_end = 999999999999999
            
            if peak_to_sec_peak_dist_end > len(signal) and inv_peak_to_sec_peak_dist_end > len(signal):
                abs_min = np.argmin(peak_to_peak_signal_end)
            else:
                abs_min = 9999999999999

            ends[peak_center] = peak_center + min(peak_to_sec_peak_dist_end, inv_peak_to_sec_peak_dist_end, abs_min)

        
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

            if peak_to_sec_peak_dist_start > len(signal) and inv_peak_to_sec_peak_dist_start > len(signal):
                abs_min = np.argmin(peak_to_peak_signal_start)
            else:
                abs_min = 9999999999999

            starts[peak_center] = peak_center - min(peak_to_sec_peak_dist_start, inv_peak_to_sec_peak_dist_start, abs_min)

            # end edge    
            if len(secondary_peak_centers_end) >= 1:
                peak_to_sec_peak_dist_end = min(secondary_peak_centers_end)
            else:
                peak_to_sec_peak_dist_end = 999999999999999

            if len(inv_secondary_peak_centers_end) >= 1:
                inv_peak_to_sec_peak_dist_end = min(inv_secondary_peak_centers_end)
            else:
                inv_peak_to_sec_peak_dist_end = 999999999999999
                
            if peak_to_sec_peak_dist_end > len(signal) and inv_peak_to_sec_peak_dist_end > len(signal):
                abs_min = np.argmin(peak_to_peak_signal_end)
            else:
                abs_min = 9999999999999

            ends[peak_center] = peak_center + min(peak_to_sec_peak_dist_end, inv_peak_to_sec_peak_dist_end, abs_min)

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

            if peak_to_sec_peak_dist_start > len(signal) and inv_peak_to_sec_peak_dist_start > len(signal):
                abs_min = np.argmin(peak_to_peak_signal_start)
            else:
                abs_min = 9999999999999

            starts[peak_center] = peak_center - min(peak_to_sec_peak_dist_start, inv_peak_to_sec_peak_dist_start, abs_min)

            # end edge    
            if len(secondary_peak_centers_end) >= 1:
                peak_to_sec_peak_dist_end = min(secondary_peak_centers_end)
            else:
                peak_to_sec_peak_dist_end = 999999999999999

            if len(inv_secondary_peak_centers_end) >= 1:
                inv_peak_to_sec_peak_dist_end = min(inv_secondary_peak_centers_end)
            else:
                inv_peak_to_sec_peak_dist_end = 999999999999999

            if peak_to_sec_peak_dist_end > len(signal) and inv_peak_to_sec_peak_dist_end > len(signal):
                abs_min = np.argmin(peak_to_end_signal)
            else:
                abs_min = 999999999

            ends[peak_center] = peak_center + min(peak_to_sec_peak_dist_end, inv_peak_to_sec_peak_dist_end, abs_min)

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
    
    specifs = {
        "starts": starts,
        "ends": ends
    }

    return intensities, specifs

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

    intensities, specifs = get_intensities(all_peak_centers, signals, int(ladder_specifs["center"]), top_peak_distances)
    
    areas = [intensities[top_peak_distances[0]], intensities[top_peak_distances[1]], intensities[top_peak_distances[2]]]

    a, b = np.polyfit(areas, np.exp(np.array(weights))-1, 1)

    x_line = np.linspace(min(areas), max(areas))
    y_line = a * x_line + b

    '''
    plt.plot(x_line, y_line, color="red")
    plt.scatter(areas, np.exp(np.array(weights))-1)
    plt.xlabel("area under peak")
    plt.ylabel("weight [ng]")
    plt.title("area under peak - weight\nstandard curve")
    plt.show()
    '''
    return a, b

def get_sample_weight(all_peak_centers, signals, pixel_line, a, b):
    intensities, specifs = get_intensities(all_peak_centers, signals, pixel_line)
    weights = {}
    for peak_center in intensities.keys():
        weight_estimation = np.log1p(a * intensities[peak_center] + b)
        if weight_estimation < 1:
            weight = "< 1"
        else:
            weight = weight_estimation
        weights[peak_center] = weight
    return weights, specifs

def get_band_weights(preprocessed_img, ladder2=False):
    grayscale_img = (preprocessed_img * 255).astype(np.uint8) if np.issubdtype(preprocessed_img.dtype, np.floating) else preprocessed_img.copy()
    img_height, img_width = grayscale_img.shape
    # získanie signálov
    signals = get_signals_list(grayscale_img, img_width)

    # pozície peakov
    all_peak_centers = {}
    for index, signal in enumerate(signals):
        all_peak_centers[index] = get_peak_centers(signals, index, 5, 20)

    # nájdenie ladderu a lane
    all_lane_specifs, ladder_specifs = get_lane_and_ladder_specifs(width_threshold=25, min_ladder_bands=3, all_peak_centers=all_peak_centers)

    # get variables of the area->weight function and create the stand. curve
    a, b = mk_weight_fnc(ladder_specifs, all_peak_centers, signals, [45, 95, 97], "output")

    # odstránenie druhého ladderu ak ladder2=True
    if ladder2 and len(all_lane_specifs) > 0:
            all_lane_specifs.popitem()

    # priprava pre text
    text_size = cv2.getTextSize("-----", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    color_img = cv2.cvtColor(grayscale_img, cv2.COLOR_GRAY2RGB)

    results_text = ""

    for lane in all_lane_specifs:
        pixel_line = int(all_lane_specifs[lane]["center"])

        sample_weights, specifs = get_sample_weight(all_peak_centers, signals, pixel_line, a, b)

        results_text += f"lane {lane}:\n"
        for i, peak_center in enumerate(sample_weights.keys()):
            sample_weight = sample_weights[peak_center]
            results_text += f"{sample_weight} ng\n"
            start = specifs["starts"][peak_center]
            end = specifs["ends"][peak_center]

            # pridanie textu na obrázok
            lane_center = int(all_lane_specifs[lane]["center"])
            centered_x_position = lane_center - text_size[0] // 2
            y_position = int(peak_center)

            if np.isnan(sample_weight):
                continue

            if isinstance(sample_weight, (int, float)):
                weight_to_print = round(sample_weight, 1)
            else:
                weight_to_print = sample_weight

            cv2.putText(color_img, f"{weight_to_print} ng", (centered_x_position, 15 + 15*i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.line(color_img, (lane_center, start), (lane_center, end), (255, 0, 0), 2)
            cv2.putText(color_img, "_", (lane_center, start),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(color_img, "_", (lane_center, end),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        results_text += "--------------\n"

        weights = [45, 95, 97]
        centered_x_ladder_position = int(ladder_specifs["center"] - text_size[0] // 2)
        ladder_center_signal = signals[int(ladder_specifs["center"])]
        distances, properties = sig.find_peaks(ladder_center_signal, height=5, distance = 5, prominence=5)
        slope, intercept, main_ladder_band_dist = get_sample_size_function_variables(signals, ladder_specifs, [1517, 1000, 517])
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

        intensities, specifs = get_intensities(all_peak_centers, signals, int(ladder_specifs["center"]), top_peak_distances)

        for i, y_band_position in enumerate(main_ladder_band_dist):
            start = specifs["starts"][y_band_position]
            end  = specifs["ends"][y_band_position]

            cv2.putText(color_img, f"{weights[i]}", (centered_x_ladder_position, 15 + 15*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.line(color_img, (centered_x_ladder_position, start), (centered_x_ladder_position, end), (255, 0, 0), 2)
            cv2.putText(color_img, "_", (centered_x_ladder_position, start),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(color_img, "_", (centered_x_ladder_position, end),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    return color_img, results_text

def get_sample_concentrations(preprocessed_img, threshold, volume, ladder2):
    grayscale_img = (preprocessed_img * 255).astype(np.uint8) if np.issubdtype(preprocessed_img.dtype, np.floating) else preprocessed_img.copy()
    img_height, img_width = grayscale_img.shape
    # získanie signálov
    signals = get_signals_list(grayscale_img, img_width)

    # pozície peakov
    all_peak_centers = {}
    for index, signal in enumerate(signals):
        all_peak_centers[index] = get_peak_centers(signals, index, 5, 20)

    # nájdenie ladderu a lane
    all_lane_specifs, ladder_specifs = get_lane_and_ladder_specifs(width_threshold=25, min_ladder_bands=3, all_peak_centers=all_peak_centers)

    # odstránenie druhého ladderu ak ladder2=True
    if ladder2 and len(all_lane_specifs) > 0:
            all_lane_specifs.popitem()

    a, b = mk_weight_fnc(ladder_specifs, all_peak_centers, signals, [45, 95, 97], "output")

    text_size = cv2.getTextSize("--.-", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    color_img = cv2.cvtColor(grayscale_img, cv2.COLOR_GRAY2RGB)

    results_text = ""

    
    cv2.putText(color_img, "concentration [ng/microliter]", (len(signals) // 2 , 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1, cv2.LINE_AA)

    for lane in all_lane_specifs:
        pixel_line = int(all_lane_specifs[lane]["center"])
        lane_center = int(all_lane_specifs[lane]["center"])
        centered_x_position = lane_center - text_size[0] // 2
        
        # threshold signal
        clear_signal = []
        for value in signals[pixel_line]:
            if value <= threshold:
                clear_signal.append(0)
            else:
                clear_signal.append(float(value))
        abs_weight = np.log1p(a * sum(clear_signal) + b)

        sample_concentration = abs_weight / volume

        if np.isnan(sample_concentration):
            continue
        conc_to_print = round(sample_concentration, 1)
 
        
        results_text += f"lane {lane}: \n"
        results_text += f"{sample_concentration} ng/µl\n--------------\n"

        cv2.putText(color_img, f"{conc_to_print}", (centered_x_position, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1, cv2.LINE_AA)

    return color_img, results_text


# funcitons for report
def img_to_base64(img):
    # convert numpy array image to base64 PNG.
    buf = BytesIO()
    # convert floats
    if np.issubdtype(img.dtype, np.floating):
        img = (img * 255).astype(np.uint8)
    plt.imsave(buf, img, cmap="gray", format="png")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def generate_report(output_html_path,
                    original_img=None,
                    preprocess_params=None,
                    preprocessed_img=None,
                    annotated_size_img=None,
                    size_results_text=None,
                    annotated_conc_img=None,
                    conc_results_text=None,
                    annotated_weight_img=None,
                    weight_results_text=None,
                    sample_volume=None):

    original_b64 = img_to_base64(original_img) if original_img is not None else None
    preprocessed_b64 = img_to_base64(preprocessed_img) if preprocessed_img is not None else None
    annotated_size_b64 = img_to_base64(annotated_size_img) if annotated_size_img is not None else None
    annotated_conc_b64 = img_to_base64(annotated_conc_img) if annotated_conc_img is not None else None
    annotated_weight_b64 = img_to_base64(annotated_weight_img) if annotated_weight_img is not None else None

    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Gel Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h2 { color:#003366; border-bottom: 2px solid #003366; padding-bottom: 4px; }
            .img-box { margin: 20px 0; text-align:center; }
            img { max-width: 700px; border:1px solid #ccc; border-radius:6px; }
            pre { background:#f4f4f4; padding:10px; border-radius:6px; }
        </style>
    </head>
    <body>

        <h1>Gel Electrophoresis Report</h1>

        {% if original_b64 or preprocessed_b64 %}
        <h2>Original vs Preprocessed</h2>

        <div style="display: flex; gap: 20px; justify-content: center; margin-bottom: 30px;">
            
            {% if original_b64 %}
            <div style="text-align: center;">
                <h3>Original</h3>
                <img src="data:image/png;base64,{{ original_b64 }}" style="max-width: 400px;">
            </div>
            {% endif %}

            {% if preprocessed_b64 %}
            <div style="text-align: center;">
                <h3>Preprocessed</h3>
                <img src="data:image/png;base64,{{ preprocessed_b64 }}" style="max-width: 400px;">
            </div>
            {% endif %}

        </div>
        {% endif %}

        {% if annotated_size_b64 %}
        <h2>Annotated Image - Size</h2>
        <div style="text-align: center;">
            <img src="data:image/png;base64,{{ annotated_size_b64 }}" style="max-width: 500px;">
        </div>
        {% endif %}

        {% if size_results %}
        <h2>Size Analysis</h2>
        <pre>{{ size_results }}</pre>
        {% endif %}

        {% if annotated_conc_b64 %}
        <h2>Annotated Image - Concentration</h2>
        <p><b>Used sample volume:</b> {{ sample_volume }} µL</p>
        <div style="text-align: center;">
            <img src="data:image/png;base64,{{ annotated_conc_b64 }}" style="max-width: 500px;">
        </div>
        {% endif %}

        {% if conc_results %}
        <h2>Concentration Analysis</h2>
        <pre>{{ conc_results }}</pre>
        {% endif %}

        {% if annotated_weight_b64 %}
        <h2>Annotated Image - Band Weights</h2>
        <div style="text-align: center;">
            <img src="data:image/png;base64,{{ annotated_weight_b64 }}" style="max-width: 500px;">
        </div>
        {% endif %}

        {% if weight_results %}
        <h2>Weight Estimation</h2>
        <pre>{{ weight_results }}</pre>
        {% endif %}

        {% if preprocess_params %}
        <h2>Preprocessing Parameters</h2>
        <pre>{{ preprocess_params }}</pre>
        {% endif %}

    </body>
    </html>
    """


    template = Template(html_template)

    html = template.render(
        preprocess_params=preprocess_params,
        original_b64=original_b64,
        preprocessed_b64=preprocessed_b64,
        annotated_size_b64=annotated_size_b64,
        size_results=size_results_text,
        annotated_conc_b64=annotated_conc_b64,
        conc_results=conc_results_text,
        annotated_weight_b64=annotated_weight_b64,
        weight_results=weight_results_text,
        sample_volume=sample_volume
    )

    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html)

def generate_txt_report(output_txt_path,
                        preprocess_params=None,
                        size_results_text=None,
                        conc_results_text=None,
                        weight_results_text=None,
                        sample_volume=None):
    report = ""

    report += "=== PREPROCESSING PARAMETERS ===\n"
    report += f"{preprocess_params}\n\n" if preprocess_params else "Not performed.\n\n"

    report += "=== SIZE ANALYSIS ===\n"
    report += f"{size_results_text}\n\n" if size_results_text else "Not performed.\n\n"

    report += "=== CONCENTRATION ANALYSIS ===\n"
    if conc_results_text:
        report += f"Sample volume: {sample_volume} µL\n"
        report += f"{conc_results_text}\n\n"
    else:
        report += "Not performed.\n\n"

    report += "=== BAND WEIGHT ESTIMATION ===\n"
    report += f"{weight_results_text}\n\n" if weight_results_text else "Not performed.\n\n"

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(report)

