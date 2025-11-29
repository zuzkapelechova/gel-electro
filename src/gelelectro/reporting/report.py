import base64
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from jinja2 import Template


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