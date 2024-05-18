import os
import shutil
import json
import torch
import numpy as np
import ultralytics.utils.plotting
from ultralytics import YOLO
from PIL import Image
from torchvision import models, transforms
from torch import nn
from tqdm import tqdm
from collections import defaultdict
import json
from config_file import load_config, update_config, save_config


class ImageObjectDetector:
    """
        Класс для обработки изображений моделями Yolo и кастомной моделью классификации
    """

    def __init__(self,
                 image_source: str,
                 image_result: str = None,
                 imagejs_result: str = None,
                 model: str = None,
                 type_of_task_else_default_models: str = "det",
                 iou: float = 0.5,
                 conf: float = 0.5,
                 augment: bool = False,
                 half: bool = False,
                 imgsz: tuple = (512, 512),
                 path_to_temp: str = None,
                 class_mapping: defaultdict = None,
                 timestamp: str = None,
                 device: str = None):

        if device is None or device == 'default':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(f"{device}")

        self.source = image_source
        self.results_dict = {}
        self.config_data = load_config()
        self.update_config = update_config

        if class_mapping is None:
            path_to_config = os.path.join(self.config_data["script_path"], "./config.json")
            with open(path_to_config, "r", encoding="utf-8") as js:
                self.class_mapping = json.load(js)["class_mapping"]
        else:
            self.class_mapping = class_mapping

        # Инициализация кастомной модели классификации
        self.model_class = models.vit_b_16(weights=None)
        num_features = self.model_class.heads[0].in_features
        self.model_class.heads[0] = nn.Linear(num_features, 3)
        path_to_weights = os.path.join(self.config_data["script_path"], "./weights/weights_class/vit_b_16.pt")
        self.model_class.load_state_dict(torch.load(path_to_weights, map_location=self.device)["model_state_dict"])
        self.model_class.to(self.device)  # Перенос модели на устройство
        self.model_class.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.iou = iou
        self.conf = conf
        self.imgsz = imgsz
        self.augment = augment
        self.half = half
        self.model = model
        self.task = type_of_task_else_default_models

        path_to_temp = self.config_data["script_path"] if path_to_temp is None else path_to_temp
        # image_result = self.config_data["script_path"] if image_result is None else image_result
        imagejs_result = self.config_data["script_path"] if imagejs_result is None else imagejs_result
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") if timestamp is None else timestamp
        # self.path_to_image_result = os.path.join(image_result, "ann")
        self.path_to_imagejs_result = os.path.join(imagejs_result, "js")
        self.path_to_temp = os.path.join(path_to_temp, "temp")
        # self.path_to_ann_time_dir = os.path.join(self.path_to_image_result, timestamp)
        self.path_to_js_time_dir = os.path.join(self.path_to_imagejs_result, f"{timestamp}_image")

        self.__exists__([self.path_to_temp,
                         # self.path_to_image_result,
                         self.path_to_imagejs_result,
                         # self.path_to_ann_time_dir,
                         self.path_to_js_time_dir])

        self.__get_model__()
        self.__start_process__()
        self.__del_temp__(self.path_to_temp)

    @staticmethod
    def __del_temp__(directory_path):
        if os.path.exists(directory_path):
            shutil.rmtree(directory_path)

    def __exists__(self, paths: list = None) -> None:
        if paths:
            for index, path in enumerate(paths):
                if not os.path.exists(path):
                    os.makedirs(path)

    def save_results_to_json(self, image_filename, results_dict):
        json_filename = os.path.splitext(image_filename)[0]
        json_path = os.path.join(self.path_to_js_time_dir, f"{json_filename}.json")
        with open(json_path, 'w', encoding="utf-8") as json_file:
            json.dump(results_dict, json_file, ensure_ascii=False, indent=4)

    def __get_model__(self):
        if self.model:
            self.model = YOLO(fr"{self.model}")
        else:
            if self.task == "det":
                self.model = YOLO(r'../weights/weights_detect/yolov8n.pt')
            elif self.task == "seg":
                self.model = YOLO(r'../weights/weights_detect/yolov8n-seg.pt')
            else:
                raise KeyError("Не верный ввод, доступны задачи: det, seg")

        self.model.to(self.device)  # Перенос модели YOLO на устройство

    def pred_class(self, image):
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model_class(image)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()

    def __start_process__(self):
        results = self.model.predict(source=self.source,
                                     imgsz=self.imgsz,
                                     conf=self.conf,
                                     iou=self.iou,
                                     augment=self.augment,
                                     half=self.half)

        for ri, sample in enumerate(results):

            detect = sample.boxes
            orig_img = sample.orig_img
            boxes = detect.xyxy.cpu().numpy()

            class_predictions = []
            label_predictions = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cropped_img = orig_img[y1:y2, x1:x2]
                cropped_img_pil = Image.fromarray(cropped_img)
                class_prediction = self.pred_class(cropped_img_pil)
                label = self.class_mapping[str(class_prediction[0])]
                class_predictions.append(class_prediction[0])
                label_predictions.append(label)

            classes = torch.tensor(class_predictions, device=detect.cls.device)
            labels = label_predictions

            try:
                segment = sample.masks
                masks = segment.xy
                masks = [array.tolist() for array in masks]
            except Exception:
                masks = None

            name_frame = os.path.basename(sample.path)

            self.results_dict = {
                "name_frame": name_frame,
                "classes": classes.tolist(),
                "labels": labels,
                "boxes": boxes.tolist(),
                "masks": masks,
                "track_history": None,
                "distance": None
            }

            """
            path = os.path.join(self.path_to_ann_time_dir, os.path.basename(sample.path))
            im_bgr = sample.plot(conf=False, labels=False)
            im_rgb = Image.fromarray(im_bgr[..., ::-1])
            im_rgb.save(path)
            """

            self.save_results_to_json(name_frame, self.results_dict)
