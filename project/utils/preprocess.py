from pathlib import Path

import numpy as np
from PIL import Image


def preprocess_image(image_path: str | Path) -> np.ndarray:
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise ValueError("Invalid image file.") from exc

    image = image.resize((224, 224))
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array
