from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')
    model.train(
        data="archive/dataset.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        cache="ram",
        workers=4,
        exist_ok=True,
        project="F:/Python/ComputerVision/lab7/results",
    )


if __name__ == '__main__':
    main()
