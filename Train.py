from library.GraduateModel import GraduateModel
from library.MetricsVisualizer import MetricsVisualizer
from ultralytics import YOLO
from torchvision import models

"""exclude_model = ["get_weight", "convnext_base", "convnext_large", "convnext_small",
                 "densenet161", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
                 "efficientnet_v2_l", "efficientnet_v2_m", "regnet_x_8gf", "regnet_x_32gf",
                 "regnet_y_32gf", "regnet_y_128gf", "resnext101_32x8d", "vit_h_14", "vit_l_16",
                 "vit_l_32", "googlenet", "inception_v3", "squeezenet1_0", "squeezenet1_1"]

optimizer_list = ["SGD", "SGD", "SGD", "SGD", "SGD", "AdamW", "AdamW", "AdamW", "AdamW",
                  "AdamW", "AdamW", "AdamW", "AdamW", "AdamW", "AdamW", "AdamW", "AdamW", "AdamW",
                  "SGD", "SGD", "SGD", "SGD", "SGD", "SGD", "SGD", "SGD", "SGD", "SGD", "SGD", "SGD", "SGD",
                  "SGD", "SGD", "SGD", "SGD", "SGD", "AdamW", "AdamW", "AdamW", "AdamW", "SGD", "SGD", "SGD",
                  "SGD", "SGD", "SGD", "SGD", "SGD", "SGD", "SGD", "SGD", "SGD", "SGD", "SGD", "SGD"]


# Получение списка доступных моделей
model_list = sorted(name for name in models.__dict__
                    if name.islower() and not name.startswith("__")
                    and name not in exclude_model
                    and callable(models.__dict__[name]))
print(model_list)"""

"""for index, model in enumerate(model_list):

    train = GraduateModel(name_model_user="",
                          name_model=f"{model}",
                          path_to_data="./data",
                          path_to_weights="./weights",
                          path_to_metrics_train="./metrics_train",
                          path_to_metrics_test="./metrics_test",
                          is_use_imagenet_weights=True,
                          name_optimizer=optimizer_list[index],
                          num_epochs=10,
                          batch_size=10,
                          train_size_img=(224, 224))
    train.graduate()"""

"""# Построение графиков
metrics_visualizer = MetricsVisualizer()"""

"""train = GraduateModel(name_model_user="good_bad_mask",
                      name_model="vit_b_16",
                      path_to_data="./data",
                      path_to_weights="./weights",
                      path_to_metrics_train="./metrics_train_mask",
                      path_to_metrics_test="./metrics_test_mask",
                      is_use_imagenet_weights=True,
                      name_optimizer="SGD",
                      num_epochs=30,
                      batch_size=10,
                      train_size_img=(224, 224),
                      is_gray=True)

train.graduate()"""

"""# Построение графиков
metrics_visualizer = MetricsVisualizer(path_to_metrics_train="./metrics_train_mask",
                                       path_to_metrics_test="./metrics_test_mask",
                                       path_to_save_plots="./mask_plot")"""


def write_labels(results):
    output_directory = ("C:/Users/NightMare/PycharmProjects/cv-classification-segmentation-nanoparticles/"
                        "data/data_detection_train_SUPER/labels")

    for index, sample in enumerate(results):
        path_to_file = results[index].path

        # Получение имени файла без расширения
        name_file = os.path.splitext(os.path.basename(path_to_file))[0]

        path = os.path.join(output_directory, f"{name_file}.txt")

        detect = results[index].boxes

        # Преобразование class_ в числовой тип данных
        classes = detect.cls.cpu().numpy().astype(int)

        # Преобразование boxes_ в числовой тип данных и учет устройства
        boxes_ = detect.xywhn.cpu().numpy()

        with open(path, "w") as txt_file:
            for i in range(len(classes)):
                txt_file.write(f"{classes[i]} {boxes_[i][0]} {boxes_[i][1]} {boxes_[i][2]} {boxes_[i][3]}\n")
            print(f"файл {name_file} записан")


def main():

    # model = YOLO('yolov8m.yaml')
    model = YOLO('yolov8n.yaml').load('./weights/weights_detect/yolov8n.pt')

    """# Load custom weights
    model = YOLO(r'C:/Users/NightMare/PycharmProjects/cv-classification-segmentation-nanoparticles/'
                 r'minibbox_detect/yolov8m/weights/last.pt')"""

    # Train
    """model.train(data="C:/Users/NightMare/PycharmProjects/DeerAI/data_coco_mask/deer_ai/deer_ai.yaml",
                batch=8,
                imgsz=512,
                save=True,
                device=0,
                project="deer_detect",
                name="yolov8n",
                exist_ok=True,
                optimizer="AdamW",
                verbose=False,
                seed=42,
                cos_lr=True,
                val=True,
                plots=True,
                workers=4,
                epochs=1000,
                resume=True)"""

    # Test
    """results = model.predict(source="C:/Users/NightMare/PycharmProjects/cv-classification-segmentation-nanoparticles/"
                                   "data/data_classification_test/image/",
                            device=0,
                            imgsz=512,
                            save=True,
                            conf=0.05,
                            iou=0.05,
                            show_labels=True,
                            augment=True)"""

    # write_labels(results=results)


if __name__ == "__main__":
    main()
