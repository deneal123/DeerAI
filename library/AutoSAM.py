import os
from tqdm import tqdm
from ultralytics.models.sam import Predictor as SAMPredictor
from ultralytics.utils.plotting import plot_images
from PIL import Image


# Создаем SAMPredictor
overrides = dict(conf=0.25, task='segment', mode='predict', imgsz=512, model="sam_b.pt")
predictor = SAMPredictor(overrides=overrides)

# Создаем словарь для хранения bbox координат по пути к изображениям
image_bbox_dict = {}

path_to_labels = "C:/Users/NightMare/PycharmProjects/DeerAI/data_coco_mask/roe_deer_labels"
path_to_images = "C:/Users/NightMare/PycharmProjects/DeerAI/data_coco_mask/roe_deer"

# Выбор класса
class_id = 0  # Здесь выберите желаемый класс
output_dir = "C:/Users/NightMare/PycharmProjects/DeerAI/data_coco_mask/roe_masks_yolo"

# Проверяем существует ли директория, если нет, создаем ее
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

for name in os.listdir(path_to_labels):
    label_path = os.path.join(path_to_labels, name)
    image_path = os.path.join(path_to_images, name[:-4] + ".png")  # Assuming images have .jpg extension

    if os.path.exists(image_path):
        if image_path not in image_bbox_dict:
            image_bbox_dict[image_path] = []

        with open(label_path, "r") as label:
            lines = label.readlines()

        for line in lines:
            values = [float(val) for val in line.strip().split()]
            image_bbox_dict[image_path].append(values[1:])  # Add bbox coordinates to the corresponding image


# Преобразовать координаты пары в последовательность координат
def flatten_coordinates(coords):
    flattened_coords = []
    for x, y in coords:
        flattened_coords.extend([x, y])
    return flattened_coords


for image_path, bboxes in tqdm(image_bbox_dict.items(), total=len(image_bbox_dict)):
    predictor.set_image(image_path)

    # Преобразуем нормированные координаты в абсолютные и изменяем формат на x1y1x2y2
    abs_bboxes = []
    for bbox in bboxes:
        # Преобразуем координаты из формата xywh в x1y1x2y2
        x, y, w, h = bbox
        x1 = int((x - w / 2) * 512)
        y1 = int((y - h / 2) * 512)
        x2 = int((x + w / 2) * 512)
        y2 = int((y + h / 2) * 512)
        abs_bboxes.append([x1, y1, x2, y2])

    results = predictor(bboxes=abs_bboxes)

    # Получаем имя изображения без расширения файла
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Формируем имя файла
    filename = f"{image_name}.txt"
    filepath = os.path.join(output_dir, filename)

    # Открываем файл для записи
    with open(filepath, "w") as file:

        # Сохраняем полигоны в текстовых файлах
        for idx, mask in enumerate(results[0].masks.xyn):

            # Преобразуем координаты и записываем полигоны в текстовый файл
            flattened_coords = flatten_coordinates(mask)
            file.write(f"{class_id} {' '.join(map(str, flattened_coords))}\n")

    # Сбрасываем изображение
    predictor.reset_image()
