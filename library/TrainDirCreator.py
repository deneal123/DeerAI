import os
import random
import shutil
from tqdm import tqdm


class TrainDirCreator:

    def __init__(self,
                 image_dirs: list,
                 dest_dir: str,
                 ratio=(60, 20, 20)):
        self.dirs = image_dirs
        self.ratio = ratio

        """
        ratio[0] = train
        ratio[1] = test
        ratio[2] = valid
        """

    def create_train_dirs(self, output_dir="data"):
        # Создание основной директории
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Создание поддиректорий для train, test и validation
        for dataset_type in ["train_dataset", "test_dataset", "valid_dataset"]:
            dataset_path = os.path.join(output_dir, dataset_type)
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)

        # Перемешивание и распределение изображений
        for image_dir in self.dirs:
            class_name = os.path.basename(image_dir)
            print(f"\nClass_name: {class_name}")
            image_list = os.listdir(image_dir)
            random.shuffle(image_list)
            total_images = len(image_list)
            train_count = int(total_images * self.ratio[0] / 100)
            print(f"Train_count: {train_count}")
            test_count = int(total_images * self.ratio[1] / 100)
            print(f"Test_count: {test_count}")
            valid_count = int(total_images * self.ratio[2] / 100)
            print(f"Valid_count: {valid_count}")

            # Перемешивание и копирование изображений в директории
            for i, image_name in enumerate(tqdm(image_list, desc="Shutil images", unit="image")):
                src_path = os.path.join(image_dir, image_name)
                if i < train_count:
                    dest_path = os.path.join(output_dir, "train_dataset", class_name, image_name)
                elif i < train_count + test_count:
                    dest_path = os.path.join(output_dir, "test_dataset", class_name, image_name)
                else:
                    dest_path = os.path.join(output_dir, "valid_dataset", class_name, image_name)

                # Создание директории для класса, если ее нет
                if not os.path.exists(os.path.dirname(dest_path)):
                    os.makedirs(os.path.dirname(dest_path))

                shutil.copy(src_path, dest_path)


def get_path_to_dir(path_to_data):
    list_dir = os.listdir(path_to_data)
    train_dir = os.path.join(path_to_data, list_dir[0])
    test_dir = os.path.join(path_to_data, list_dir[1])
    valid_dir = os.path.join(path_to_data, list_dir[2])
    return train_dir, test_dir, valid_dir


# Пример использования
if __name__ == "__main__":
    image_dirs = ["C:/Users/NightMare/PycharmProjects/DeerAI/deer",
                  "C:/Users/NightMare/PycharmProjects/DeerAI/garb"]
    dest_dir = "./"
    """train_dir_creator = TrainDirCreator(image_dirs,
                                        image_dirs)
    train_dir_creator.create_train_dirs()"""
