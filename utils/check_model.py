"""
Diagnostic script — prints key metadata about a YOLO .pt model.

Usage:
    python -m utils.check_model                     # uses default path from config
    python -m utils.check_model --model ./best.pt   # explicit path
"""
import argparse
from ultralytics import YOLO


def check_model(model_path: str) -> None:
    model = YOLO(model_path)
    print(f"Model path  : {model_path}")
    print(f"Training imgsz: {model.overrides.get('imgsz', 'default (640)')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO model diagnostics")
    parser.add_argument("--model", type=str, default="./input_models/best.pt", help="Path to .pt model file")
    args = parser.parse_args()
    check_model(args.model)
