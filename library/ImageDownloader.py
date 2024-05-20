from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import os
from urllib.parse import urljoin
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import requests
import torch
from torchvision import models
import torchvision.transforms as transforms
from mmocr.apis import MMOCRInferencer
from ScriptPath import get_script_path


class ImageDownloader:
    """
    Класс для загрузки и фильтрации изображений из Интернета.

    Методы:
        __init__: Инициализация модели ResNet и OCR модели, а также параметров фильтров.
        brightness: Вычисление яркости изображения.
        has_text: Проверка наличия текста на изображении.
        is_real_image: Проверка, является ли изображение реальным.
        download_images_from_urls: Загрузка и фильтрация изображений по заданным параметрам.
        get_image_urls: Получение URL-адресов изображений из поискового запроса.
        parser_img: Парсинг и загрузка изображений с применением фильтров.
    """

    def __init__(self,
                 query: str = "Бабочка",
                 num_images: int = 1,
                 downl_dir: str = "./parser_img",
                 apply_resnet_filter: str = False,
                 apply_text_filter: str = False,
                 apply_brightness_filter: str = False,
                 apply_size_filter: str = False,
                 apply_format_filter: str = False,
                 min_brightness: int = 30,
                 max_brightness: int = 220,
                 min_width: int = 80,
                 min_height: int = 80,
                 allowed_formats=('JPEG', 'PNG', 'JPG')):
        """
        Инициализация модели ResNet и OCR модели, а также параметров фильтров.

        Параметры:
            apply_resnet_filter: Применять фильтр ResNet для определения реальности изображения.
            apply_text_filter: Применять фильтр для удаления изображений с текстом.
            apply_brightness_filter: Применять фильтр для яркости изображения.
            apply_size_filter: Применять фильтр для размера изображения.
            apply_format_filter: Применять фильтр для формата изображения.
            min_brightness: Минимальная яркость пикселей.
            max_brightness: Максимальная яркость пикселей.
            min_width: Минимальная ширина изображения.
            min_height: Минимальная высота изображения.
            allowed_formats: Разрешенные форматы изображений.
        """
        # Загрузка предварительно обученной модели ResNet
        self.model = models.resnet18(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, 2)
        path_to_weights = os.path.join(get_script_path(), "./weights/model.pt")
        self.model.load_state_dict(torch.load(path_to_weights))
        self.model.eval()

        # Загрузка OCR модели
        self.reader = MMOCRInferencer(det='DBNet', rec='ABINet')

        # Основные параметры
        self.query = query
        self.num_images = num_images
        self.download_path = downl_dir

        # Параметры фильтров
        self.apply_resnet_filter = apply_resnet_filter
        self.apply_text_filter = apply_text_filter
        self.apply_brightness_filter = apply_brightness_filter
        self.apply_size_filter = apply_size_filter
        self.apply_format_filter = apply_format_filter
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_width = min_width
        self.min_height = min_height
        self.allowed_formats = allowed_formats

    def __str__(self):
        return f"Apply ResNet Filter: {self.apply_resnet_filter}\n" \
               f"Apply Text Filter: {self.apply_text_filter}\n" \
               f"Apply Brightness Filter: {self.apply_brightness_filter}\n" \
               f"Apply Size Filter: {self.apply_size_filter}\n" \
               f"Apply Format Filter: {self.apply_format_filter}"

    def brightness(self, image):
        """
        Вычисление средней яркости пикселей изображения.

        Параметры:
            image: Изображение в формате PIL.

        Возвращает:
            float: Средняя яркость пикселей изображения.
        """
        pixel_brightness = image.convert("L").getdata()
        return sum(pixel_brightness) / len(pixel_brightness)

    def has_text(self, image_path):
        """
        Проверка наличия текста на изображении.

        Параметры:
            path_to_image: Путь к изображению.

        Возвращает:
            bool: True, если на изображении есть текст, иначе False.
        """
        result = self.reader(image_path, save_vis=False, return_vis=False)
        res = result['predictions'][0].get('rec_texts')[0]
        return len(res) > 0

    def is_real_image(self, image_path):
        """
        Проверка, является ли изображение реальным (а не мультяшным).

        Параметры:
            image_path: Путь к изображению.

        Возвращает:
            bool: True, если изображение является реальным, иначе False.
        """
        input_image = Image.open(image_path).convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)[0]
        return probabilities.item() > 0.5

    def download_images_from_urls(self, image_urls):
        """
        Загрузка и фильтрация изображений по заданным параметрам.

        Параметры:
            image_urls: Список URL-адресов изображений.
            download_path: Путь для сохранения загруженных изображений.
            query: Поисковый запрос (название темы).
        """
        if not os.path.exists(self.download_path):
            os.makedirs(self.download_path)

        # Download and save the images
        for i, url in enumerate(tqdm(image_urls, desc="Downloading images", unit="image")):
            try:
                # Send a request to download the image
                image_response = requests.get(url)
                if image_response.status_code == 200:
                    # Open the image using PIL
                    img = Image.open(BytesIO(image_response.content))
                    img_path = os.path.join(self.download_path, f"{self.query}_{i}.png")

                    # Check brightness of the image
                    if self.apply_brightness_filter:
                        img_brightness = self.brightness(img)
                        if img_brightness < self.min_brightness or img_brightness > self.max_brightness:
                            continue

                    # Check image dimensions
                    if self.apply_size_filter:
                        img_width, img_height = img.size
                        if img_width < self.min_width or img_height < self.min_height:
                            continue

                    # Check image format
                    if self.apply_format_filter:
                        img_format = img.format.upper()
                        if img_format not in self.allowed_formats:
                            continue

                    img = img.resize(size=(512, 512))
                    img = img.convert("RGB")
                    img.save(img_path)

                    # Check if the image is real if the filter is enabled
                    if self.apply_resnet_filter:
                        if not self.is_real_image(img_path):
                            os.remove(img_path)  # Remove cartoonish images

                    # Check for text in the image if the filter is enabled
                    if self.apply_text_filter:
                        if self.has_text(img_path):
                            os.remove(img_path)  # Remove images with text

            except Exception as e:
                continue

    def get_image_urls(self):
        """
        Получение URL-адресов изображений из поискового запроса.

        Параметры:
            query: Поисковый запрос (название темы).
            num_images: Количество изображений для загрузки.

        Возвращает:
            list: Список URL-адресов изображений.
        """
        # Create an instance of ChromeOptions
        chrome_options = Options()

        # Create an instance of ChromeDriver with options
        driver = webdriver.Chrome(options=chrome_options)

        driver.get(f"https://www.google.com/search?q={self.query}&tbm=isch")

        # Scroll the page to load more images
        last_height = driver.execute_script("return document.body.scrollHeight")
        while len(driver.find_elements(By.TAG_NAME, 'img')) < self.num_images:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(4)  # Wait for the page to load
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        # Extract image URLs
        image_urls = [img.get_attribute('src') for img in driver.find_elements(By.TAG_NAME,
                                                                               'img')][:self.num_images]

        driver.quit()
        return image_urls

    def parser_img(self):
        """
        Парсинг и загрузка изображений с применением фильтров.

        Параметры:
            query: Поисковый запрос (название темы).
            num_images: Количество изображений для загрузки.
            downl_dir: Путь для сохранения загруженных изображений.
        """
        # Get the image URLs
        print(self.__str__())
        # Download and save the images
        self.download_images_from_urls(self.get_image_urls())


if __name__ == "__main__":
    img_downloader = ImageDownloader(query="Слон",
                                     num_images=10)
    # img_downloader.parser_img()
