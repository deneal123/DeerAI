import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.models as models
from library.TrainDirCreator import get_path_to_dir
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary
import re


class GraduateModel:

    def __init__(self,
                 name_model_user: str = "",
                 name_model: str = "efficientnet_b0",
                 path_to_data: str = "./data",
                 path_to_weights: str = "./weights",
                 path_to_metrics_train: str = "./metrics_train",
                 path_to_metrics_test: str = "./metrics_test",
                 is_use_imagenet_weights: bool = True,
                 name_optimizer: str = "Adam",
                 num_epochs: int = 3,
                 batch_size: int = 32,
                 train_size_img: tuple = (224, 224),
                 is_gray: str = False):
        self.name_model = name_model
        self.name_model_user = name_model_user if name_model_user != "" else self.name_model
        self.path_to_data = path_to_data
        directories_to_create = [path_to_weights, path_to_metrics_train, path_to_metrics_test]
        self.create_directories_if_not_exist(*directories_to_create)
        self.path_to_weights = os.path.join(path_to_weights, f"{self.name_model_user}.pt")
        self.path_to_metrics_train = os.path.join(path_to_metrics_train, f"{self.name_model_user}.pt")
        self.path_to_metrics_test = os.path.join(path_to_metrics_test, f"{self.name_model_user}.pt")
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.size_img = train_size_img
        self.name_optimizer = name_optimizer
        self.size_img = train_size_img
        self.model = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.classes = None
        self.total_samples = None
        self.targets = None
        self.class_counts = None
        self.class_weights = None
        self.criterion = None
        self.scheduler = None
        self.imagenet = is_use_imagenet_weights
        self.is_gray = is_gray

        # Получение списка доступных моделей
        self.model_list = sorted(name for name in models.__dict__
                                 if name.islower() and not name.startswith("__")
                                 and name != "get_weight"
                                 and callable(models.__dict__[name]))

        # Перемещение модели на GPU, если CUDA доступен
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def graduate(self):
        # Получаем трансформер
        self.get_transform()
        # Получаем генераторы обучения, валидации и теста
        self.get_loaders()
        # Загружаем модель
        self.get_model()
        # Получаем веса классов
        self.get_classes_weights()
        # Определяем оптимизатор, функцию потерь и планировщик
        self.get_opt_crit_sh()
        # Вывод архитектуры
        """summary(self.model,
                (3, self.size_img[0], self.size_img[1]),
                self.batch_size)"""
        # Выводим информацию
        print(self.__str__())
        # Обучаем
        self.train_model()
        # Тестируем
        self.evaluate_model()

    def __str__(self):

        return f"""
        
        ----------------------------------------------------------------------
        
        Определенное устройство: {self.device}
        Количество эпох обучения {self.num_epochs}
        Размер пакета: {self.batch_size}
        Размер изображений для обучения: {self.size_img}
        Выбранная модель: {self.name_model}
        Пользовательское название модели: {self.name_model_user}
        Модель сохранена по пути: {self.path_to_weights}
        Метрики тренировки сохранены по пути: {self.path_to_metrics_train}
        Метрики теста сохранены по пути: {self.path_to_metrics_test}
        Данные загружены из директории: {self.path_to_data}
        Классы: {self.classes}
        Количество примеров каждого класса: {self.class_counts}
        Веса каждого класса: {self.class_weights}
        Выбранный оптимизатор: {self.name_optimizer}
        
        ----------------------------------------------------------------------
        
        """

    def create_directories_if_not_exist(self, *directories):
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)

    # Функция для показа изображения
    def image_show(self, img):
        img = img / 2 + 0.5  # unnormalize
        np_img = img.cpu().numpy()
        plt.imshow(np.transpose(np_img, (1, 2, 0)))
        plt.show()

    # Функция для сохранения модели
    def save_model(self):
        # Сохраняем модель
        torch.save({
            'epoch': self.num_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()},
            f"{self.path_to_weights}")

    def save_metrics_train(self,
                           train_loss_values,
                           valid_loss_values,
                           f1_values):
        metrics = {
            'train_loss': train_loss_values,
            'valid_loss': valid_loss_values,
            'valid_f1': f1_values
        }
        # Сохранение метрик
        torch.save(metrics, self.path_to_metrics_train)

    def save_metrics_test(self,
                          f1,
                          class_acc_dir):
        metric = {
            'f1_value': f1,
            'Acc_dir': class_acc_dir
        }
        # Сохранение метрик
        torch.save(metric, self.path_to_metrics_test)

    def get_transform(self):
        if self.is_gray:
            self.transform = {
                "train": transforms.Compose([
                    transforms.Resize((self.size_img[0], self.size_img[1])),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop(size=(self.size_img[0], self.size_img[1])),
                    transforms.ToTensor()
                ]),
                "valid": transforms.Compose([
                    transforms.Resize((self.size_img[0], self.size_img[1])),
                    transforms.ToTensor()
                ])
            }
        else:
            self.transform = {
                "train": transforms.Compose([
                    transforms.Resize((self.size_img[0], self.size_img[1])),
                    transforms.AutoAugment(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]),
                "valid": transforms.Compose([
                    transforms.Resize((self.size_img[0], self.size_img[1])),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            }

    # Функция для загрузки данных
    def get_loaders(self):
        # Получаем пути к директориям
        train_dir, test_dir, valid_dir = get_path_to_dir(f"{self.path_to_data}")
        # Создание датасетов
        train_dataset = ImageFolder(train_dir, transform=self.transform["train"])
        test_dataset = ImageFolder(test_dir, transform=self.transform["valid"])
        valid_dataset = ImageFolder(valid_dir, transform=self.transform["valid"])
        # Создание загрузчиков данных
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        self.valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
        # Список классов
        self.classes = train_dataset.classes
        self.total_samples = len(train_dataset)
        self.targets = train_dataset.targets

    def get_model(self):
        # Проверка доступности модели в torchvision
        if self.name_model in self.model_list:
            # Проверка доступности весов ImageNet
            if self.imagenet:
                try:
                    if self.name_model in ["inception_v3", "googlenet"]:
                        # Попытка загрузки модели с весами ImageNet
                        self.model = models.__dict__[self.name_model](init_weights=True)
                    else:
                        if self.name_model in ["regnet_y_128gf", "vit_h_14"]:
                            self.model = models.__dict__[self.name_model](weights="IMAGENET1K_SWAG_E2E_V1")
                        else:
                            self.model = models.__dict__[self.name_model](weights="IMAGENET1K_V1")
                except KeyError:
                    # Если веса ImageNet для данной модели недоступны, загрузка без весов
                    print(
                        f"Предобученные веса ImageNet для модели {self.name_model} недоступны."
                        f" Загрузка модели без предобученных весов.")
                    if self.name_model in ["inception_v3", "googlenet"]:
                        # Попытка загрузки модели с весами ImageNet
                        self.model = models.__dict__[self.name_model](init_weights=False)
                    else:
                        self.model = models.__dict__[self.name_model](weights=None)
            else:
                # Если не указано использование весов ImageNet, загрузить модель без них
                if self.name_model in ["inception_v3", "googlenet"]:
                    self.model = models.__dict__[self.name_model](init_weights=False)
                else:
                    self.model = models.__dict__[self.name_model](weights=None)

            try:
                name_without_numbers = re.sub(r'\d+', '', self.name_model)
                # Проверка доступности слоя classifier
                if hasattr(self.model, 'classifier'):
                    if name_without_numbers == "densenet":
                        num_features = self.model.classifier.in_features
                        self.model.classifier = nn.Linear(num_features, len(self.classes))
                    elif name_without_numbers == "squeezenet_":
                        num_features = 512
                        self.model.classifier[-1] = nn.Linear(num_features, len(self.classes))
                    else:
                        num_features = self.model.classifier[-1].in_features
                        self.model.classifier[-1] = nn.Linear(num_features, len(self.classes))
                elif hasattr(self.model, 'last_linear'):
                    num_features = self.model.last_linear.in_features
                    self.model.last_linear = nn.Linear(num_features, len(self.classes))
                elif hasattr(self.model, 'fc'):
                    num_features = self.model.fc.in_features
                    self.model.fc = nn.Linear(num_features, len(self.classes))
                elif hasattr(self.model, 'head'):
                    num_features = self.model.head.in_features
                    self.model.head = nn.Linear(num_features, len(self.classes))
                elif hasattr(self.model, "heads"):
                    num_features = self.model.heads[0].in_features
                    self.model.heads[0] = nn.Linear(num_features, len(self.classes))
                else:
                    print(f"Слой classifier не найден в модели. name_model:{self.name_model}")
            except Exception as ex:
                print(f"Exception: {ex}, name_model: {self.name_model}")

            # Перемещение модели на GPU и указание устройства
            self.model = self.model.to(self.device)
        else:
            print(f"Модель {self.name_model} не найдена в torchvision."
                  f"\nСписок доступных моделей: {self.model_list}")

    def get_classes_weights(self):
        # Получение количества примеров для каждого класса
        self.class_counts = np.bincount(self.targets)
        # Вычисление весов классов
        self.class_weights = torch.tensor([self.total_samples / count for count in self.class_counts],
                                          dtype=torch.float)
        self.class_weights = self.class_weights.to(self.device)

    def get_opt_crit_sh(self):
        # Определение функции потерь с учетом весов классов
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        self.optimizer = optim.__dict__[f"{self.name_optimizer}"](self.model.parameters(), lr=0.001)
        # Создание планировщика LR
        # ReduceLROnPlateau уменьшает скорость обучения, когда метрика перестает уменьшаться
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=2, verbose=True)

    # Функция для обучения модели с валидацией
    def train_model(self):

        train_loss_values = []
        valid_loss_values = []
        f1_values = []

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, labels in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs} (Train)",
                                       unit="sample"):
                inputs, labels = inputs.cuda(), labels.cuda()
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            # Вычисление loss на валидационном датасете и метрик
            self.model.eval()
            valid_loss = 0.0
            best_f1 = 0.0
            all_predictions = []
            all_labels = []
            with torch.no_grad():
                for inputs, labels in tqdm(self.valid_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs} (Eval)",
                                           unit="sample"):
                    inputs, labels = inputs.cuda(), labels.cuda()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    valid_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / len(self.train_loader.dataset)
            valid_loss = valid_loss / len(self.valid_loader.dataset)

            accuracy = accuracy_score(all_labels, all_predictions)
            precision = precision_score(all_labels, all_predictions, average='weighted')
            recall = recall_score(all_labels, all_predictions, average='weighted')
            f1 = f1_score(all_labels, all_predictions, average='weighted')

            # Сообщаем планировщику LR о текущей ошибке на валидационном наборе
            self.scheduler.step(valid_loss)

            # we want to save the model if the accuracy is the best
            if f1 > best_f1:
                self.save_model()
                best_f1 = f1

            # Добавление значений метрик в списки
            train_loss_values.append(epoch_loss)
            valid_loss_values.append(valid_loss)
            f1_values.append(f1)

            # Сохранение метрик
            self.save_metrics_train(train_loss_values,
                                    valid_loss_values,
                                    f1_values)

            print(f"\nEpoch {epoch + 1}/{self.num_epochs}, Training Loss: {epoch_loss}, Validation Loss: {valid_loss}")
            print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")

        print("Тренировка завершена!")

    # Функция для оценки модели на тестовом датасете
    def evaluate_model(self):
        self.model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []

        # Initialize variables to track correct predictions for each class
        class_correct = [0] * len(self.classes)
        class_total = [0] * len(self.classes)

        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc="Test", unit="sample"):
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Calculate class-wise correct predictions
                for i in range(len(labels)):
                    label = labels[i].item()
                    class_correct[label] += (predicted[i] == labels[i]).item()
                    class_total[label] += 1

        accuracy = correct / total
        print(f"Test Accuracy: {accuracy}")

        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        print(f"Precision: {precision}, Recall: {recall}, F1-score: {f1}")

        class_acc_dir = {}
        # Print accuracy for each class
        for i in range(len(self.classes)):
            class_acc = 100 * class_correct[i] / class_total[i] if class_total[i] != 0 else 0
            class_acc_dir[self.classes[i]] = class_acc
            print('Accuracy of %5s : %2d %%' % (self.classes[i], class_acc))

        self.save_metrics_test(f1,
                               class_acc_dir)
