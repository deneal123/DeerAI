from ultralytics import YOLO

# Load custom weights
model = YOLO(r'../weights/weights_detect/best.pt')

results = model.predict(source="C:/Users/NightMare/PycharmProjects/DeerAI/data_hack/roe_deer/",
                        device=0,
                        imgsz=512,
                        save=True,
                        conf=0.3,
                        iou=0.3,
                        show_labels=True,
                        augment=True,
                        save_crop=True)
