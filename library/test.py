import os
import json

folder_path = "./js/2024_05_18_16_43_04_image/"


def get_prob_for_class(folder_path, deer_count, musk_deer_count, capreolus_count):
    deer_probs = []
    musk_deer_probs = []
    capreolus_probs = []
    for filename in os.listdir(folder_path):
        deer_count_i = 0
        musk_deer_count_i = 0
        capreolus_count_i = 0
        if filename.endswith(".json"):
            with open(os.path.join(folder_path, filename), "r") as f:
                data = json.load(f)
                for label in data['labels']:
                    if label == "Олень":
                        deer_count_i += 1
                    elif label == "Кабарга":
                        musk_deer_count_i += 1
                    elif label == "Косуля":
                        capreolus_count_i += 1
        if deer_count != 0 :
            deer_probs.append(deer_count_i/deer_count)
        if musk_deer_count != 0 :
            musk_deer_probs.append(musk_deer_count_i/musk_deer_count)
        if capreolus_count != 0 :
            capreolus_probs.append(capreolus_count_i/capreolus_count)
        
    return deer_probs, musk_deer_probs, capreolus_probs
        

def get_info_from_image(folder_path):
    deer_count = 0
    musk_deer_count = 0
    capreolus_count = 0

    # Читаем и объединяем данные из JSON файлов
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            with open(os.path.join(folder_path, filename), "r") as f:
                data = json.load(f)
                for label in data['labels']:
                    if label == "Олень":
                        deer_count += 1
                    elif label == "Кабарга":
                        musk_deer_count += 1
                    elif label == "Косуля":
                        capreolus_count += 1
    print(f"Количество объектов класса 'Олень': {deer_count}")
    print(f"Количество объектов класса 'Кабарга': {musk_deer_count}")
    print(f"Количество объектов класса 'Косуля': {capreolus_count}")
    p1, p2, p3 = get_prob_for_class(folder_path, deer_count, musk_deer_count, capreolus_count)
    print(p1, p2, p3)
    
    
def get_info_from_video(folder_path):
    deer_count = 0
    musk_deer_count = 0
    capreolus_count = 0

    # Читаем и объединяем данные из JSON файлов
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            with open(os.path.join(folder_path, filename), "r") as f:
                data = json.load(f)
                ids = []
                for frames in data:
                    if data[frames]['track_ids:'] is not None:
                        for i in range(len(data[frames]['track_ids:'])):
                            if data[frames]['track_ids:'][i] not in ids:
                                ids.append(data[frames]['track_ids:'][i])
                                for label in data[frames]['labels']:
                                        if label == "Олень":
                                            deer_count += 1
                                        elif label == "Кабарга":
                                            musk_deer_count += 1
                                        elif label == "Косуля":
                                            capreolus_count += 1
    print(f"Количество объектов класса 'Олень': {deer_count}")
    print(f"Количество объектов класса 'Кабарга': {musk_deer_count}")
    print(f"Количество объектов класса 'Косуля': {capreolus_count}")
    
get_info_from_image(folder_path)