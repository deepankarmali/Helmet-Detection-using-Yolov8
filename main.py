from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.yaml")
    model.to('cuda') 

    results = model.train(data="config.yaml", epochs=200)

if __name__ == '__main__':
    main()