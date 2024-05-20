import os
import shutil
import pytube
import youtube_dl
import re
from PIL import Image
from tqdm import tqdm
import math
from moviepy.editor import VideoFileClip
import torch
from torchvision import models
import torchvision.transforms as transforms
from ScriptPath import get_script_path


class VideoCutter:

    def __init__(self,
                 download_path: str = "./temp",
                 frame_path: str = "./parser_video",
                 fps: int or float = 24,
                 size_frame: tuple = (512, 512),
                 apply_resnet_filter: bool = True):
        self.download_path = download_path
        self.frame_path = frame_path
        self.fps = fps
        self.vid_title = None
        self.size = size_frame

        # Загрузка предварительно обученной модели ResNet
        self.model = models.resnet18(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, 2)
        path_to_weights = os.path.join(get_script_path(), "./weights/model.pt")
        self.model.load_state_dict(torch.load(path_to_weights))
        self.model.eval()

        # Параметры фильтров
        self.apply_resnet_filter = apply_resnet_filter

    def get_clean_title(self, title):
        """
        Удаляет из названия видео регулярные знаки и пробелы.

        Параметры:
            title (str): Название видео.

        Возвращает:
            str: Название видео без регулярных знаков и пробелов.
        """
        clean_title = re.sub(r'[^\w\s]', '', title)  # Удаляем все символы, кроме букв, цифр и пробелов
        return clean_title

    def is_real_image(self, input_image):
        """
        Проверка, является ли изображение реальным (а не мультяшным).

        Параметры:
            image_path: Путь к изображению.

        Возвращает:
            bool: True, если изображение является реальным, иначе False.
        """
        input_image = input_image.convert('RGB')
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

    def download_video_pytube(self, video_url):
        """
        Скачивает видео из YouTube по ссылке и сохраняет его в указанную директорию,
        используя библиотеку pytube.

        Параметры:
            video_url (str): Ссылка на YouTube видео.
        """
        if not os.path.exists(self.download_path):
            os.makedirs(self.download_path)

        yt = pytube.YouTube(video_url)
        stream = yt.streams.get_highest_resolution()
        self.vid_title = self.get_clean_title(yt.title)
        video_filename = f"{self.vid_title}.mp4"
        video_path = os.path.join(self.download_path, video_filename)
        stream.download(output_path=self.download_path, filename=video_filename)

        return video_path

    def download_video_youtubedl(self, video_url):
        """
        Скачивает видео из YouTube по ссылке и сохраняет его в указанную директорию,
        используя библиотеку youtube_dl.

        Параметры:
            video_url (str): Ссылка на YouTube видео.
        """

        if not os.path.exists(self.download_path):
            os.makedirs(self.download_path)

        ydl_opts = {'outtmpl': os.path.join(self.download_path, 'video.mp4')}
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=True)
            title = info_dict.get('title', 'video')
            self.vid_title = self.get_clean_title(title)  # Сохраняем общее название для видео

        video_path = os.path.join(self.download_path, self.get_clean_title(title) + '.mp4')
        path = os.path.join(self.download_path, 'video.mp4')
        shutil.move(path, video_path)

        return video_path

    def download_video(self, video_url):
        """
        Скачивает видео из YouTube по ссылке и сохраняет его в указанную директорию,
        используя библиотеку pytube или youtube_dl.

        Параметры:
            video_url (str): Ссылка на YouTube видео.
        """
        try:
            return self.download_video_pytube(video_url)
        except (pytube.exceptions.RegexMatchError, pytube.exceptions.ExtractError):
            return self.download_video_youtubedl(video_url)

    def cut_video(self, video_path, segments):
        """
        Нарезает видео на кадры в соответствии с указанными временными сегментами.

        Параметры:
            video_path (str): Путь к видео файлу.
            segments (list): Список кортежей временных сегментов в формате (начало, конец).
        """
        print(video_path)
        clip = VideoFileClip(video_path)
        for i, segment in enumerate(tqdm(segments, desc="Slice segments", unit="video")):
            start_time, end_time = segment
            clipped = clip.subclip(start_time, end_time)
            # Изменение размера кадров
            clipped_resized = clipped.resize(self.size)

            frame_folder = os.path.join(self.frame_path)
            if not os.path.exists(frame_folder):
                os.makedirs(frame_folder)

            # Вычисляем шаг, с которым нужно сохранять кадры в зависимости от self.fps
            frame_step = max(math.floor(clipped_resized.fps / self.fps), 1)

            # Итерация по кадрам с использованием frame_step
            for j, frame in enumerate(
                    tqdm(clipped_resized.iter_frames(), desc="Processing frames", unit="frame", position=1)):
                if j % frame_step == 0:
                    frame_image = Image.fromarray(frame)

                    # Check if the image is real if the filter is enabled
                    if self.apply_resnet_filter:
                        if not self.is_real_image(frame_image):
                            continue

                    frame_path = os.path.join(frame_folder, f"{self.vid_title}_frame{i:04d}_{j:04d}.png")
                    frame_image.save(frame_path)

        clip.close()

        # Удаление временной папки с видео
        shutil.rmtree(self.download_path)


# Пример использования
if __name__ == "__main__":
    cutter = VideoCutter(download_path="./temp",
                         frame_path="./deer_1",
                         fps=2,
                         size_frame=(512, 512))
    """video_path = cutter.download_video(video_url="https://www.youtube.com/watch?v=-EaPM0tMDbw")
    cutter.cut_video(video_path=video_path,
                     segments=[(60, 2340)])"""
