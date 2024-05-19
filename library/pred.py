import os
import csv
import torch
from torchvision import models, transforms
from PIL import Image, ImageFile
from torch import nn

# Разрешаем загрузку поврежденных изображений
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Инициализация кастомной модели классификации
model_class = models.swin_t(weights=None)
num_features = model_class.head.in_features
model_class.head = nn.Linear(num_features, 3)
path_to_weights = "../weights/weights_class/swin_t.pt"
model_class.load_state_dict(torch.load(path_to_weights)["model_state_dict"])
model_class.cuda()
model_class.eval()

# Словарь для преобразования классов животных в цифры
class_to_num = {"Кабарга": 0, "Косуля": 1, "Олень": 2}

# Словарь, который преобразует предсказания модели в нужный формат
model_to_class = {1: 0, 2: 1, 0: 2}

# Преобразования для изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# Функция для предсказания класса изображения
def predict_image_class(image_path, model, transform):
    try:
        image = Image.open(image_path).convert('RGB')  # Конвертируем изображение в RGB
        image = transform(image).unsqueeze(0).cuda()

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)

        return predicted.item()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return 2  # Принять класс "Олень" (2 в модели) в случае ошибки


# Функция для обработки изображений в директории
def process_images(directory, model, transform):
    results = []

    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            file_path = os.path.join(directory, filename)

            # Предсказать класс изображения
            predicted_class = predict_image_class(file_path, model, transform)

            # Преобразовать предсказанный класс в нужный формат
            class_num = model_to_class[predicted_class]

            # Сохранить результат
            results.append({"img_name": filename, "class": class_num})

    return results


# Функция для сохранения результатов в CSV файл
def save_to_csv(results, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['img_name', 'class']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)


# Указание директории с изображениями и имени выходного CSV файла
directory = "C:/Users/NightMare/Downloads/test_minprirodi_Parnokopitnie"
output_file = "C:/Users/NightMare/Downloads/prischlidrinkcofee_submission.csv"

# Обработка изображений и сохранение результатов в CSV файл
results = process_images(directory, model_class, transform)
save_to_csv(results, output_file)
