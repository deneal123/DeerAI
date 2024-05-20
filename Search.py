from library.ImageDownloader import ImageDownloader
from library.VideoSlicer import VideoCutter


"""query = [
    'Car'
]

for q in query:
    img_downloader = ImageDownloader(query=q,
                                     num_images=1000,
                                     downl_dir="./deer___",
                                     apply_text_filter=True,
                                     apply_format_filter=True,
                                     apply_brightness_filter=True,
                                     apply_size_filter=True,
                                     apply_resnet_filter=True)
    img_downloader.parser_img()"""


"""cutter = VideoCutter(download_path="./temp",
                     frame_path="./deer___",
                     fps=1/4,
                     size_frame=(512, 512),
                     apply_resnet_filter=True)
video_path = cutter.download_video(video_url="https://www.youtube.com/watch?v=o_zrFDH7rBQ")
cutter.cut_video(video_path=video_path,
                 segments=[(40, 450)])"""
