from library.TrainDirCreator import TrainDirCreator
from library.DataFarmer import DataFarmer


image_dirs = ["C:/Users/NightMare/PycharmProjects/DeerAI/data_coco_mask/1_mask_deer",
              "C:/Users/NightMare/PycharmProjects/DeerAI/data_coco_mask/mask_deer_bad"]
dest_dir = "./"
train_dir_creator = TrainDirCreator(image_dirs,
                                    dest_dir)
# train_dir_creator.create_train_dirs()


# farmer = DataFarmer(input_dir="C:/Users/NightMare/PycharmProjects/DeerAI/data_coco_mask")
# farmer.brightness_sort(threshold_white=210, threshold_black=2)
# farmer.good_bad_mask_sort()
# farmer.find_missmatch(dir_="C:/Users/NightMare/PycharmProjects/DeerAI/data_coco_mask/deer_masks_yolo")
# farmer.save_coco_txt(number_class=2)
# farmer.convert_format()
# farmer.resize_images()
# farmer.draw_bounding_boxes(annotations_folder="C:/Users/NightMare/PycharmProjects/DeerAI/data_coco_mask/roe_deer_labels")
# farmer.draw_masks(annotations_folder="C:/Users/NightMare/PycharmProjects/DeerAI/data_coco_mask/deer_masks_yolo")
# farmer.rewrite_cls_txt(int_class=2)
# farmer.yolo_data()
