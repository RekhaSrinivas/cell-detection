from ultralytics import YOLO
import argparse

def main(args):
    model = YOLO("yolov8n.pt")  # or path to your custom model
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch
    )
    print("Training completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data.yaml')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    args = parser.parse_args()
    main(args)
