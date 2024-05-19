import argparse
from library.VideoObjectTracker import VideoObjectTracker
from library.ImageObjectDetector import ImageObjectDetector
from library.SourceChecker import SourceChecker
from datetime import datetime
import torch

parser = argparse.ArgumentParser(description='PipeLine для обработки изображений и видео')
parser.add_argument('--source', type=str,
                    default='./source', help='Путь к директории источника')
parser.add_argument('--model_weight', type=str,
                    default=None, help='Список путей к весам моделей')
parser.add_argument('--iou', type=float,
                    default=0.5, help='Насколько сильно могут перекрываться объекты')
parser.add_argument('--conf', type=float,
                    default=0.5, help='Пороговую уверенность')
parser.add_argument('--augment', type=bool,
                    default=False, help='Замедление, увеличение точности за счет аффинных преобразований')
parser.add_argument('--half', type=bool,
                    default=False, help='Ускорение, незначительная потеря в точности за счет float16 (вместо float32)')
parser.add_argument('--color_track', type=bytearray,
                    default=(0, 191, 255), help='Цвет трека')
parser.add_argument('--thickness_track', type=int,
                    default=3, help='Толщина линии трека')
parser.add_argument('--len_track', type=int,
                    default=30, help='Длина трека в кадрах')
parser.add_argument('--show_track', type=bool,
                    default=False, help='Показать трек?')
parser.add_argument('--view_img_heatmap', type=bool,
                    default=True, help='накладывать heatmap на кадр?')
parser.add_argument('--shape_heatmap', type=str,
                    default="circle", help='Форма heatmap')
parser.add_argument('--decay_factor_heatmap', type=float,
                    default=0.95, help='Коэффициент затухания')
parser.add_argument('--heatmap_alpha', type=float,
                    default=0.5, help='Прозрачность heatmap')
parser.add_argument('--line_width_eye', type=int,
                    default=1, help='Ширина линии глаза')
parser.add_argument('--pixel_per_meter', type=int,
                    default=10, help='Количество пикселей на 1 метр')
parser.add_argument('--show_distance', type=bool,
                    default=False, help='Показывать дистанцию до объекта?')
parser.add_argument('--vid_stride', type=int,
                    default=5, help='Сколько пропускать кадров при обработке видео?')
parser.add_argument('--device', type=str,
                    default=None, help='Какое устройство использовать cuda/cpu?')


# Функция для обработки данных изображений или видео
def PipeLine(source: str = "./source",
             model_weight: str = None,
             iou: float = 0.5,
             conf: float = 0.5,
             augment: bool = False,
             half: bool = False,
             color_track: bytearray = (0, 191, 255),
             thickness_track: int = 1,
             len_track: int = 100,
             show_track: bool = False,
             view_img_heatmap: bool = True,
             shape_heatmap: str = "circle",
             decay_factor_heatmap: float = 0.95,
             heatmap_alpha: float = 0.5,
             line_width_eye: int = 1,
             pixel_per_meter: int = 10,
             show_distance: bool = False,
             vid_stride: int = 5,
             device: str = None) -> None:

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    BGR = (color_track[2], color_track[1], color_track[0])
    that = SourceChecker(source)

    if device is None:
        device = 'default'
    else:
        device = f"{device}"

    if that is None:
        raise FileExistsError("Директория пуста")

    if model_weight is None:
        model_weight = "./weights/weights_detect/best.pt"

    if that == 'video':
        track = VideoObjectTracker(video_source=source,
                                   model_weight=model_weight,
                                   iou=iou,
                                   conf=conf,
                                   augment=augment,
                                   half=half,
                                   BGR_track=BGR,
                                   thickness_track=thickness_track,
                                   len_track=len_track,
                                   show_track=show_track,
                                   view_img_heatmap=view_img_heatmap,
                                   shape_heatmap=shape_heatmap,
                                   decay_factor_heatmap=decay_factor_heatmap,
                                   heatmap_alpha=heatmap_alpha,
                                   line_width_eye=line_width_eye,
                                   pixel_per_meter=pixel_per_meter,
                                   show_distance=show_distance,
                                   timestamp=timestamp,
                                   vid_stride=vid_stride,
                                   device=device)
    elif that == "image":
        annotate = ImageObjectDetector(image_source=source,
                                       model_weight=model_weight,
                                       iou=iou,
                                       conf=conf,
                                       augment=augment,
                                       half=half,
                                       timestamp=timestamp,
                                       device=device)
    elif that is None:
        print("Загружен не правильный формат изображения или видео.")


if __name__ == "__main__":
    args = parser.parse_args()
    PipeLine(source=args.source,
             model_weight=args.model_weight,
             iou=args.iou,
             conf=args.conf,
             augment=args.augment,
             half=args.half,
             color_track=args.color_track,
             thickness_track=args.thickness_track,
             len_track=args.len_track,
             show_track=args.show_track,
             view_img_heatmap=args.view_img_heatmap,
             shape_heatmap=args.shape_heatmap,
             decay_factor_heatmap=args.decay_factor_heatmap,
             heatmap_alpha=args.heatmap_alpha,
             line_width_eye=args.line_width_eye,
             pixel_per_meter=args.pixel_per_meter,
             show_distance=args.show_distance,
             vid_stride=args.vid_stride,
             device=args.device)
