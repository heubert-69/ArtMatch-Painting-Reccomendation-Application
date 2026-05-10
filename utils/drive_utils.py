import requests
import io
import numpy as np
from PIL import Image


def download_image_as_array(file_id):
    session = requests.Session()

    url = "https://drive.google.com/uc?export=download&id=" + file_id

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = session.get(url, headers=headers, stream=True)

    if response.status_code != 200:
        raise Exception(f"Drive download failed: {response.status_code}")

    content_type = response.headers.get("Content-Type", "")

    if "text/html" in content_type:
        raise ValueError("Got HTML instead of image → check file permissions or invalid file_id")

    img = Image.open(io.BytesIO(response.content)).convert("RGB")

    return np.array(img)