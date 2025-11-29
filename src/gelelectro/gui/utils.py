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

    if ladder2:
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
            sample_size = estimate_sample_size(slope, intercept, sample_distance)
            sample_sizes.append(sample_size)

            results_text += f"{str(sample_size)}bp\n"

            # add image annotations
            centered_x_position = int(all_lane_specifs[lane]["center"]) - text_size[0] // 2
            y_position = int(sample_distance)
            cv2.putText(color_img, f"{int(sample_size)}bp", (centered_x_position, 15 + 15*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
            cv2.putText(color_img, "-----", (centered_x_position, sample_distance), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)

        results_text += "--------------\n"

    centered_x_ladder_position = int(ladder_specifs["center"] - text_size[0] // 2)
    sizes = [1517, 1000, 517]
    for i, y_band_position in enumerate(main_ladder_band_dist):
        cv2.putText(color_img, f"{sizes[i]}bp", (centered_x_ladder_position, 15 + 15*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
        cv2.putText(color_img, "-----", (centered_x_ladder_position, y_band_position), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)



    return color_img, results_text
    return grayscale_img, results_text

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
                    preprocess_params,
                    original_img=None,
                    preprocessed_img=None,
                    annotated_img=None,
                    band_results=None):
    """
    Generates HTML report using data already computed in the GUI.
    Does NOT run any preprocessing or analysis.
    """

    original_b64 = img_to_base64(original_img) if original_img is not None else None
    preprocessed_b64 = img_to_base64(preprocessed_img) if preprocessed_img is not None else None
    annotated_b64 = img_to_base64(annotated_img) if annotated_img is not None else None

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

        <h2>1. Preprocessing Parameters</h2>
        <pre>{{ preprocess_params }}</pre>

        {% if original_b64 or preprocessed_b64 %}
        <h2>2. Original vs Preprocessed</h2>

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


        {% if annotated_b64 %}
        <h2>3. Annotated Image - size</h2>
        <div class="img-box">
            <img src="data:image/png;base64,{{ annotated_b64 }}">
        </div>
        {% endif %}

        {% if band_results %}
        <h2>4. Size Analysis</h2>
        <pre>{{ band_results }}</pre>
        {% endif %}

    </body>
    </html>
    """

    template = Template(html_template)

    html = template.render(
        preprocess_params=preprocess_params,
        original_b64=original_b64,
        preprocessed_b64=preprocessed_b64,
        annotated_b64=annotated_b64,
        band_results=band_results
    )

    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html)
