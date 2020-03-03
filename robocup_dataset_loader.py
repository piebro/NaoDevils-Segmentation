from pycocotools.coco import COCO
import numpy as np
import sys
import io
import os
import cv2

def get_data_list(dataset_folder, dataset_type=None, dataset_validation_size=0.2, ran_seed=42):
  folder_names = []
  for file_name in os.listdir(dataset_folder):
    if file_name.endswith(".json"):
      folder_names.append(file_name[:-5])


  img_paths_annotations = []
  for folder_name in folder_names:
    # to surpress the coco lib prints
    save_stdout = sys.stdout
    sys.stdout = io.StringIO()
    coco = COCO(os.path.join(dataset_folder, folder_name + ".json"))
    # to use the normal stdout again
    sys.stdout = save_stdout

    img_ids = sorted(coco.imgs.keys())
    for img_id in img_ids:
      annotation = coco.loadAnns(coco.getAnnIds(img_id))
      img_file_name = coco.loadImgs(img_id)[0]['file_name']
      img_path = os.path.join(folder_name, img_file_name)
      img_path = os.path.join(dataset_folder, img_path)
      img_paths_annotations.append((img_path, annotation))
  
  np.random.seed(ran_seed)
  np.random.shuffle(img_paths_annotations)
  num_of_val_img = round(len(img_paths_annotations)*dataset_validation_size)
  if(dataset_type=="training"):
    img_paths_annotations = img_paths_annotations[:-num_of_val_img]
  if(dataset_type=="validation"):
    img_paths_annotations = img_paths_annotations[-num_of_val_img:]
  
  return img_paths_annotations