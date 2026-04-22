import argparse
from pathlib import Path

from model.train import train_and_save_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train COVID-19 vs Normal chest X-ray model")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Dataset root containing COVID and Normal folders",
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir)
    output_model_path = project_root / "covid_model.h5"

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset path not found: {data_dir}")

    saved_path = train_and_save_model(
        data_dir=data_dir,
        output_model_path=output_model_path,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    print(f"Model saved to: {saved_path}")


if __name__ == "__main__":
    main()
