from pathlib import Path

import numpy as np

from utils.preprocess import preprocess_image


NORMAL_THRESHOLD = 0.50


def _sigmoid_prediction(predictions: np.ndarray) -> tuple[str, float]:
    normal_probability = float(predictions[0][0])
    if normal_probability >= NORMAL_THRESHOLD:
        return "Normal", normal_probability
    return "COVID Positive", 1.0 - normal_probability


def _softmax_prediction(predictions: np.ndarray) -> tuple[str, float]:
    probabilities = np.squeeze(predictions)
    if probabilities.ndim != 1 or probabilities.size < 2:
        raise ValueError("Unexpected model output shape.")

    class_index = int(np.argmax(probabilities))
    labels = ["COVID Positive", "Normal"]

    if class_index >= len(labels):
        raise ValueError("Model class mapping is not compatible with this application.")

    return labels[class_index], float(probabilities[class_index])


def predict_image(model, image_path: str | Path) -> tuple[str, float]:
    if model is None:
        raise RuntimeError("Model could not be loaded. Ensure covid_model.h5 is valid and available.")

    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image, verbose=0)

    if predictions.ndim == 2 and predictions.shape[1] == 1:
        return _sigmoid_prediction(predictions)

    if predictions.ndim == 2 and predictions.shape[1] >= 2:
        return _softmax_prediction(predictions)

    raise ValueError("Unsupported prediction output format from model.")
