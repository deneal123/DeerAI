import os
import json
from ScriptPath import get_script_path


class StatObjectDecider:

    def __init__(self,
                 is_image: bool,
                 imagejs_result: str = None,
                 videojs_result: str = None):

        self.is_image = is_image
        self.paths_to_json = None
        self.data = []
        self.count_image = None
        self.classes = []
        self.boxes = []
        self.masks = []

        if self.is_image:
            imagejs_result = get_script_path() if imagejs_result is None else imagejs_result
            self.path_to_imagejs_result = os.path.join(imagejs_result, "imagejs")
            self.__exists__(self.path_to_imagejs_result)
        else:
            videojs_result = get_script_path() if videojs_result is None else videojs_result
            self.path_to_videojs_result = os.path.join(videojs_result, "videojs")
            self.__exists__(self.path_to_videojs_result)

        self.__get_stat__()

    def __exists__(self, path: str = None) -> None:
        try:
            if path:
                if os.path.exists(path):
                    self.paths_to_json = [os.path.join(path, p) for p in os.listdir(path)]
        except FileExistsError as fe:
            print(fe)

    def __get_stat__(self):

        if self.is_image:
            for js in self.paths_to_json:
                with open(js) as file:
                    self.data.append(json.load(file))

            self.count_image = len(self.data)

            for sample in self.data:

                self.classes.append(sample["classes"])
                self.boxes.append(sample["boxes"])
                self.masks.append(sample["masks"])

            print(self.classes)
            print(self.boxes)
            print(self.masks)

    def __get_xy_hist(self):
        pass

