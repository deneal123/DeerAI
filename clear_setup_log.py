import os
from library.custom_logging import setup_logging
log = setup_logging()


def clear_setup_log(script_path):
    # Собираем полный путь к файлу setup.log
    script_path, _ = os.path.split(script_path)
    log_file_path = os.path.join(script_path, 'setup.log')

    try:
        # Проверяем существование файла
        if os.path.isfile(log_file_path):
            # Открываем файл в режиме записи и очищаем его
            with open(log_file_path, 'w') as log_file:
                log_file.truncate(0)
                log.info("Файл setup.log успешно очищен.")
        else:
            log.warning(f"Файл setup.log не существует в указанной директории {log_file_path}.")
    except Exception as e:
        log.error(f"Произошла ошибка при очистке файла setup.log: {e}")
