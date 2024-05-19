import os
import json
import matplotlib.pyplot as plt


class StatObjectDecider:

    def __init__(self, path_to_js: str):
        self.path_to_js = path_to_js
        self.deer_count = 0
        self.musk_deer_count = 0
        self.roe_deer_count = 0
        self.total_images = 0

    def get_info_from_image(self):
        for filename in os.listdir(self.path_to_js):
            if filename.endswith(".json"):
                self.total_images += 1
                with open(os.path.join(self.path_to_js, filename), "r") as f:
                    data = json.load(f)
                    for label in data['labels']:
                        if label == "Deer":
                            self.deer_count += 1
                        elif label == "MuskDeer":
                            self.musk_deer_count += 1
                        elif label == "RoeDeer":
                            self.roe_deer_count += 1

        return self.deer_count, self.musk_deer_count, self.roe_deer_count

    def get_info_from_video(self):
        for filename in os.listdir(self.path_to_js):
            if filename.endswith(".json"):
                with open(os.path.join(self.path_to_js, filename), "r") as f:
                    data = json.load(f)
                    ids = []
                    for frames in data:
                        if data[frames]['track_ids:'] is not None:
                            for i in range(len(data[frames]['track_ids:'])):
                                if data[frames]['track_ids:'][i] not in ids:
                                    ids.append(data[frames]['track_ids:'][i])
                                    for label in data[frames]['labels']:
                                        if label == "Deer":
                                            self.deer_count += 1
                                        elif label == "MuskDeer":
                                            self.musk_deer_count += 1
                                        elif label == "RoeDeer":
                                            self.roe_deer_count += 1
        return self.deer_count, self.musk_deer_count, self.roe_deer_count

    def plot_histogram(self):
        labels = ['Олень', 'Кабарга', 'Косуля']
        counts = [self.deer_count, self.musk_deer_count, self.roe_deer_count]

        plt.figure(figsize=(8, 6))
        plt.bar(labels, counts, color=['blue', 'green', 'red'])
        plt.xlabel('Вид оленя')
        plt.ylabel('Частота')
        plt.title('Частота появления разных видов оленей на изображениях')
        plt.show()
