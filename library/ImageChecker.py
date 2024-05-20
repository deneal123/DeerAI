import os
import shutil
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def process_file(file_path, output_directory):
    try:
        # Проверка, что файл не пустой
        if os.path.getsize(file_path) == 0:
            os.remove(file_path)
            return False

        # Открываем изображение и сохраняем его в формате PNG
        with Image.open(file_path) as img:
            new_file_path = os.path.join(output_directory, f"{os.path.splitext(os.path.basename(file_path))[0]}.png")
            img.save(new_file_path, "PNG")

        # Удаляем старый файл после конвертации
        if not file_path.lower().endswith('.png'):
            os.remove(file_path)
        return True
    except (OSError, IOError):
        # Удаляем битые файлы
        os.remove(file_path)
        return False


def process_directory(directory):
    # Шаг 1: Получаем список всех файлов в директории
    file_dict = {}
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        # Проверяем, что это файл
        if os.path.isfile(file_path):
            # Разделяем имя файла и расширение
            name, ext = os.path.splitext(filename)
            if name in file_dict:
                # Добавляем расширение к имени, если оно уже есть в словаре
                file_dict[name].append(ext)
            else:
                file_dict[name] = [ext]

    # Шаг 2: Переименовываем файлы, если у них одинаковые имена
    for name, exts in file_dict.items():
        if len(exts) > 1:
            for i, ext in enumerate(exts):
                old_file_path = os.path.join(directory, f"{name}{ext}")
                new_name = f"{name}_{i}{ext}"
                new_file_path = os.path.join(directory, new_name)
                os.rename(old_file_path, new_file_path)

    # Обновляем словарь файлов после переименования
    file_dict = {os.path.splitext(filename)[0]: [os.path.splitext(filename)[1]] for filename in os.listdir(directory) if
                 os.path.isfile(os.path.join(directory, filename))}

    # Шаг 3: Переводим все файлы в формат PNG и удаляем битые/пустые файлы
    files_to_process = [os.path.join(directory, f"{name}{ext}") for name, exts in file_dict.items() for ext in exts]

    with ThreadPoolExecutor() as executor:
        # Создаем индикатор прогресса tqdm
        with tqdm(total=len(files_to_process), desc="Processing files", unit="file") as pbar:
            futures = {executor.submit(process_file, file_path, directory): file_path for file_path in files_to_process}
            for future in as_completed(futures):
                result = future.result()
                pbar.update(1)


# Пример использования функции
process_directory('C:/Users/NightMare/PycharmProjects/DeerAI/data_hack/musk_deer')
