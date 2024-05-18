from ultralytics import YOLO
from PIL import Image
import os
from collections import defaultdict
import cv2
import numpy as np
import re
import threading
from torchvision import models, transforms
import shutil
import json
from ultralytics.solutions import heatmap
from ultralytics.utils.plotting import colors, Annotator
import math
from torch import nn
import torch
from config_file import load_config, update_config, save_config


class VideoObjectTracker:
    """
        Класс для обработки видео моделями Yolo
    """

    def __init__(self,
                 video_source: str,
                 video_result: str = None,
                 videojs_result: str = None,
                 type_of_task_else_default_models: str = "det",
                 BGR_track=(255, 191, 0),
                 thickness_track: int = 1,
                 len_track: int = 90,
                 show_track: bool = True,
                 iou: float = 0.5,
                 conf: float = 0.5,
                 augment: bool = False,
                 half: bool = False,
                 imgsz: tuple = (512, 512),
                 models_list: list = None,
                 path_to_temp: str = None,
                 view_img_heatmap: bool = True,
                 shape_heatmap: str = "circle",
                 decay_factor_heatmap: float = 1,
                 heatmap_alpha: float = 0.5,
                 line_width_eye: int = 1,
                 pixel_per_meter: int = 10,
                 show_distance: bool = True,
                 class_mapping: defaultdict = None,
                 timestamp: str = None,
                 vid_stride: int = 0,
                 device: str = None) -> None:

        if device is None or device == 'default':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(f"{device}")

        self.is_download = False
        self.paths_to_video = []
        self.names_video = []
        self.track_history = defaultdict(lambda: [])
        self.distance_history = defaultdict(lambda: [])
        self.results_dict = {}
        self.config_data = load_config()
        self.update_config = update_config
        self.vid_stride = vid_stride

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

        self.BGR = BGR_track
        self.thickness = thickness_track
        self.len_track = len_track
        self.show_track = show_track
        self.iou = iou
        self.conf = conf
        self.imgsz = imgsz
        self.augment = augment
        self.half = half
        self.models = models_list
        self.task = type_of_task_else_default_models
        self.view_img = view_img_heatmap
        self.shape = shape_heatmap
        self.decay_factor = decay_factor_heatmap
        self.heatmap_alpha = heatmap_alpha
        self.line_width_eye = line_width_eye
        self.pixel_per_meter = pixel_per_meter
        self.show_distance = show_distance

        path_to_temp = self.config_data["script_path"] if path_to_temp is None else path_to_temp
        video_result = self.config_data["script_path"] if video_result is None else video_result
        videojs_result = self.config_data["script_path"] if videojs_result is None else videojs_result
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") if timestamp is None else timestamp
        self.path_to_video_result = os.path.join(video_result, "ann")
        self.path_to_videojs_result = os.path.join(videojs_result, "js")
        self.path_to_temp = os.path.join(path_to_temp, "temp")
        self.path_to_ann_time_dir = os.path.join(self.path_to_video_result, f"{timestamp}_video")
        self.path_to_js_time_dir = os.path.join(self.path_to_videojs_result, f"{timestamp}_video")

        self.__exists__([video_source,
                         self.path_to_temp,
                         self.path_to_video_result,
                         self.path_to_videojs_result,
                         self.path_to_ann_time_dir,
                         self.path_to_js_time_dir])

        self.video_source = video_source

        self.__get_names_video__()
        self.__get_model__()
        self.__start_process__()
        self.__del_temp__(self.path_to_temp)

    @staticmethod
    def get_clean_title(title) -> str:
        clean_title = re.sub(r'[^\w\s]', '', title)
        return clean_title

    @staticmethod
    def __del_temp__(directory_path):
        if os.path.exists(directory_path):
            shutil.rmtree(directory_path)

    def save_results_to_json(self, video_filename, results_dict):
        json_filename = os.path.splitext(video_filename)[0]
        json_path = os.path.join(self.path_to_js_time_dir, f"{json_filename}.json")
        with open(json_path, 'w') as json_file:
            json.dump(results_dict, json_file, ensure_ascii=False, indent=4)

    def __get_names_video__(self):
        for path in self.paths_to_video:
            self.names_video.append(os.path.basename(path))

    def __exists__(self, paths: list = None) -> None:
        if paths:
            for index, path in enumerate(paths):
                if index == 0:
                    try:
                        self.paths_to_video = [os.path.join(path, f) for f in os.listdir(path)]
                    except Exception as ex:
                        print(ex)
                else:
                    if not os.path.exists(path):
                        os.makedirs(path)

    def pred_class(self, image):
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model_class(image)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()

    def __start_process__(self):
        try:
            threads = []
            if len(self.models) == 1:
                model = self.models[0]
                for index, video_path in enumerate(self.paths_to_video):
                    tracker_thread = threading.Thread(target=self.run_tracker_in_thread,
                                                      args=(video_path, model, index),
                                                      daemon=True)
                    tracker_thread.start()
                    threads.append(tracker_thread)
            else:
                for index, (video_path, model) in enumerate(zip(self.paths_to_video, self.models)):
                    tracker_thread = threading.Thread(target=self.run_tracker_in_thread,
                                                      args=(video_path, model, index),
                                                      daemon=True)
                    tracker_thread.start()
                    threads.append(tracker_thread)

            # Ждем завершения всех потоков
            for thread in threads:
                thread.join()
        finally:
            cv2.destroyAllWindows()

    def __get_model__(self):
        count_model = len(self.paths_to_video)
        if self.models:
            weights_list = self.models
            self.models = []
            if len(weights_list) > 1:
                for model in weights_list:
                    model = YOLO(fr"{model}")
                    model.to(self.device)  # Перенос модели YOLO на устройство
                    self.models.append(model)
            elif len(weights_list) == 1:
                for model in range(count_model):
                    model = YOLO(fr"{weights_list[0]}")
                    model.to(self.device)  # Перенос модели YOLO на устройство
                    self.models.append(model)
        else:
            if self.task == "det":
                self.models = []
                for _ in range(count_model):
                    model = YOLO(r'../weights/weights_detect/yolov8n.pt')
                    model.to(self.device)  # Перенос модели YOLO на устройство
                    self.models.append(model)
            elif self.task == "seg":
                self.models = []
                for _ in range(count_model):
                    model = YOLO(r'../weights/weights_detect/yolov8n-seg.pt')
                    model.to(self.device)  # Перенос модели YOLO на устройство
                    self.models.append(model)
            else:
                raise KeyError("Не верный ввод, доступны задачи: det, seg")

    def run_tracker_in_thread(self, filename, model, file_index):

        video = cv2.VideoCapture(filename)
        w, h, fps = (int(video.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_filename = os.path.basename(filename)
        heatmap_filename = f"heatmap_{video_filename}"

        center_point = (w//2, h)

        index = -1

        video_path = os.path.join(self.path_to_ann_time_dir, video_filename)
        heatmap_path = os.path.join(self.path_to_ann_time_dir, heatmap_filename)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
        heatmap_writer = cv2.VideoWriter(heatmap_path, fourcc, fps, (w, h))

        heatmap_obj = heatmap.Heatmap()
        heatmap_obj.set_args(colormap=cv2.COLORMAP_OCEAN,
                             imw=w,
                             imh=h,
                             view_img=self.view_img,
                             shape=self.shape,
                             classes_names=model.names,
                             decay_factor=self.decay_factor,
                             heatmap_alpha=self.heatmap_alpha)

        while True:
            index += 1

            # Пропуск кадров на основе frame_skip
            if index % self.vid_stride != 0:
                ret = video.grab()  # Grab the frame without decoding
                if not ret:
                    break
                continue

            ret, frame = video.read()

            if ret:
                results = model.track(frame,
                                      persist=True,
                                      iou=self.iou,
                                      conf=self.conf,
                                      augment=self.augment,
                                      half=self.half,
                                      imgsz=self.imgsz)

                annotator = Annotator(frame, line_width=self.line_width_eye)

                heatmap0 = heatmap_obj.generate_heatmap(frame, results)

                name_frame = f"image{index}"
                orig_shape = results[0].orig_shape
                orig_img = results[0].orig_img
                # classes = results[0].boxes.cls.cpu().numpy().astype(int)
                boxes = results[0].boxes.xywh.cpu().numpy()
                boxes_ = results[0].boxes.xyxy.cpu()

                class_predictions = []
                label_predictions = []
                for box in boxes_:
                    x1, y1, x2, y2 = map(int, box)
                    cropped_img = orig_img[y1:y2, x1:x2]
                    cropped_img_pil = Image.fromarray(cropped_img)
                    class_prediction = self.pred_class(cropped_img_pil)
                    label = self.class_mapping[str(class_prediction[0])]
                    class_predictions.append(class_prediction[0])
                    label_predictions.append(label)

                classes = torch.tensor(class_predictions, device=results[0].boxes.cls.device)
                labels = label_predictions

                try:
                    masks = results[0].masks.xy
                    masks = [array.tolist() for array in masks]
                except Exception:
                    masks = None

                frame_track_history = defaultdict(list)
                frame_distance_history = defaultdict(list)

                txt_color, txt_background, bbox_clr = ((0, 0, 0), (255, 255, 255), (255, 0, 255))

                if results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.int().cpu().tolist()

                    for box, track_id in zip(boxes_, track_ids):
                        dist = self.distance_history[track_id]

                        annotator.box_label(box, label=str(track_id), color=colors(int(track_id)))
                        annotator.visioneye(box, center_point)

                        x1, y1 = int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)  # Bounding box centroid

                        distance = (math.sqrt(
                            (x1 - center_point[0]) ** 2 + (y1 - center_point[1]) ** 2)) / self.pixel_per_meter
                        dist.append(float(distance))

                        frame_distance_history[track_id] = dist

                        if self.show_distance:
                            text_size, _ = cv2.getTextSize(f"{distance:.2f} m",
                                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            cv2.rectangle(frame,
                                          (x1, y1 - text_size[1] - 10),
                                          (x1 + text_size[0] + 10, y1),
                                          txt_background,
                                          -1)
                            cv2.putText(frame, f"{distance:.2f} m",
                                        (x1, y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5,
                                        txt_color,
                                        1)
                else:
                    track_ids = None

                res_plotted = results[0].plot(conf=False, labels=False)

                if track_ids:
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        track = self.track_history[track_id]
                        track.append((float(x), float(y)))

                        track = track[-self.len_track:]

                        frame_track_history[track_id] = track

                        # Draw the tracking lines
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))

                        if self.show_track:
                            cv2.polylines(res_plotted, [points],
                                          isClosed=False, color=self.BGR,
                                          thickness=self.thickness)

                # cv2.imshow(f"Tracking_Stream_{file_index}", res_plotted)

                video_writer.write(res_plotted)
                heatmap_writer.write(heatmap0)

                sample_dict = {
                    "name_frame": name_frame,
                    "classes": classes.tolist(),
                    "labels": labels,
                    "boxes": boxes.tolist(),
                    "masks": masks,
                    "track_history": frame_track_history,
                    "distance": frame_distance_history
                }

                self.results_dict[index] = sample_dict

                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
            else:
                break

        video.release()

        self.save_results_to_json(video_filename, self.results_dict)
