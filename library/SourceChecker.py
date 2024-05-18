import os


def SourceChecker(source):
    try:
        files = os.listdir(source)

        if files is []:
            return None

        image_files = [file for file in files
                       if file.lower().endswith(('.bmp', '.dng', '.jpeg', '.jpg', '.mpo',
                                                 '.png', '.tif', '.tiff', '.webp', '.pfm'))]
        video_files = [file for file in files
                       if file.lower().endswith(('.asf', '.avi', '.gif', '.m4v', '.mkv', '.mov',
                                                 '.mp4', '.mpeg', '.mpg', '.ts', '.wmv', '.webm'))]
        if image_files:
            return 'image'
        if video_files:
            return 'video'

    except Exception:
        return None
