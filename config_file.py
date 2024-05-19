import json
import os


# Создание стандартных настроек конфигурационного файла
def save_default_config():
    """
    Создает и сохраняет стандартные настройки конфигурационного файла.

    :param: None.
    :return: None.
    """

    config_data = {
        "class_mapping": {
            "0": "Deer",
            "1": "MuskDeer",
            "2": "RoeDeer"
        },
        "augment": False,
        "half": False,
        "view_img_heatmap": True,
        "shape_heatmap": "circle",
        "vid_stride": 5,
        "conf": 0.5,
        "decay_factor_heatmap": 0.95,
        "iou": 0.5,
        "pixel_per_meter": 10
    }

    with open('config.json', 'w') as config_file:
        json.dump(config_data, config_file, indent=4)


# Глобальная переменная для хранения предыдущих параметров
previous_config_data = None


def parse_list(self, input_str):
    try:
        # Пробуем разобрать введенную строку как JSON
        list_values = json.loads(input_str)
        # Проверяем, что полученный объект - это список
        if not isinstance(list_values, list):
            raise ValueError("Введенные данные не являются списком.")
        return list_values
    except ValueError as e:
        return None


def load_config():
    """
    Загружает данные конфигурационного файла и сохраняет их в глобальную переменную.

    :param previous_config_file: Глобальная переменная хранящая список.
    :return config_data: Список данных выгруженных из конфигурационного файла.
    """

    global previous_config_data

    # Загрузка конфигурационных параметров из файла
    with open('config.json', 'r') as config_file:
        config_data = json.load(config_file)

    # Сохраняем текущие параметры в глобальной переменной
    previous_config_data = config_data

    return config_data


def is_value_changed(key, new_value):
    """
    Создает и сохраняет стандартные настройки конфигурационного файла.

    :param previous_config_file: Глобальная переменная хранящая список.
    :return previous_config_data[key] != new_value: Возвращает в глобальную переменную...
    Новое значение для соответствующего ключа, если это значение ключа...
    Было измененно.
    :return True: Возвращает true, если значение ключа не изменилось.
    """

    global previous_config_data

    # Проверяем, изменилось ли значение параметра
    if previous_config_data is not None and key in previous_config_data:
        return previous_config_data[key] != new_value
    else:
        # Если предыдущих данных нет, считаем, что значение изменилось
        return True


def update_config(self, **kwargs):

    for key, value in kwargs.items():
        # Используем get для проверки, что ключ существует в конфигурационных данных
        if key in self.config_data and is_value_changed(key, value):
            self.config_data[key] = value

    save_config(self.config_data)


def save_config(config_data):
    """
    Сохраняет параметры в конфигурационный файл.

    :param: None.
    :return: None.
    """

    # Сохранение конфигурационных параметров в файл
    with open('config.json', 'w') as config_file:
        json.dump(config_data, config_file, indent=4)


def default_path():
    """
    Получить путь к текущему исполняемому скрипту.

    :param: None.
    :return: None.
    """

    script_path = os.path.realpath(__file__)
    script_path, _ = os.path.split(script_path)
    config_data = load_config()
    _update_config(config_data=config_data, script_path=script_path)


def _update_config(config_data, script_path=None):
    """
    Обновляет конфигурационные данные на основе переданных параметров.
    """

    if config_data is not None:
        if script_path is not None and is_value_changed('script_path', script_path):
            config_data['script_path'] = script_path

    save_config(config_data)


if __name__ == '__main__':
    save_default_config()
    default_path()
