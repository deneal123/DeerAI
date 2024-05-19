import os
import json
import csv
from collections import Counter

# Словарь для преобразования классов животных в цифры
class_to_num = {"Кабарга": 0, "Косуля": 1, "Олень": 2}

# Функция для чтения и обработки JSON файлов
def process_json_files(directory):
    results = []

    # Пройтись по всем файлам в директории
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)

            # Открыть и прочитать содержимое JSON файла
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # Извлечь имя файла и метки
                img_name = data["name_frame"]
                labels = data["labels"]

                if labels:  # Проверить, есть ли метки
                    # Подсчитать количество вхождений каждого класса
                    label_counts = Counter(labels)

                    # Найти наиболее часто встречающийся класс
                    most_common_label = label_counts.most_common(1)[0][0]

                    # Преобразовать класс в соответствующую цифру
                    class_num = class_to_num[most_common_label]
                else:
                    # Если меток нет, принять класс "Олень"
                    class_num = class_to_num["Олень"]

                # Сохранить результат
                results.append({"img_name": img_name, "class": class_num})

    return results


# Функция для сохранения результатов в CSV файл
def save_to_csv(results, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['img_name', 'class']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)


# Указание директории с JSON файлами и имени выходного CSV файла
directory = "C:/Users/NightMare/PycharmProjects/BackDeerAI/js/2024_05_19_08_14_32_image"
output_file = "C:/Users/NightMare/Downloads/prischlidrinkcofee_submission.csv"

# Обработка JSON файлов и сохранение результатов в CSV файл
results = process_json_files(directory)
save_to_csv(results, output_file)
