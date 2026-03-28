"""
Construction Safety Monitor — Training Script
"""
from ultralytics import YOLO

def train(data_yaml: str, epochs: int = 50):
    model = YOLO('yolov8s.pt')
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=16,
        name='construction_safety',
        project='runs',
        patience=10,
        save=True,
        plots=True,
    )
    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to data.yaml')
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    train(args.data, args.epochs)
