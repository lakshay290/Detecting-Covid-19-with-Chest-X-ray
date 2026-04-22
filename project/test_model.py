import argparse
from pathlib import Path

from model.load_model import get_model
from model.predict import predict_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Test COVID model prediction on a single image")
    parser.add_argument("image", type=str, help="Path to chest X-ray image")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    model_path = project_root / "covid_model.h5"

    model = get_model(model_path)
    if model is None:
        print("Error: Could not load model from project/covid_model.h5")
        return

    label, confidence = predict_image(model, args.image)
    print(f"Result: {label}")
    print(f"Confidence: {confidence * 100:.2f}%")


if __name__ == "__main__":
    main()
