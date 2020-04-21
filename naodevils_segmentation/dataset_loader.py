import sys
import io
import os
import json

import cv2
from pycocotools.coco import COCO
import numpy as np


def get_data_list(dataset_folder, dataset_validation_size=0.2, ran_seed=42):
  """
  Get a list of all training or validation img_path, annontation, meta_info
  """


  annotations_path = os.path.join(dataset_folder, "annotations")
  images_path = os.path.join(dataset_folder, "images")

  img_paths_annotations = []
  for json_file_name in os.listdir(annotations_path):
    
    # to surpress the coco lib prints
    save_stdout = sys.stdout
    sys.stdout = io.StringIO()
    coco = COCO(os.path.join(annotations_path, json_file_name))
    # to use the normal stdout again
    sys.stdout = save_stdout

    img_folder_path = os.path.join(images_path, os.path.splitext(json_file_name)[0])
    for img_id in sorted(coco.imgs.keys()):
      img_file_name = coco.loadImgs(img_id)[0]['file_name']
      img_path = os.path.join(img_folder_path, img_file_name)

      data_entry = {
        "img_path": img_path,
        "annotation": coco.loadAnns(coco.getAnnIds(img_id))
      }
      #data_entry["meta_info"] = get_image_meta_infos(img_path)
      
      img_paths_annotations.append(data_entry)
  
  np.random.seed(ran_seed)
  np.random.shuffle(img_paths_annotations)
  num_of_val_img = round(len(img_paths_annotations)*dataset_validation_size)

  dataset = {
      "training": img_paths_annotations[:-num_of_val_img],
      "validation": img_paths_annotations[-num_of_val_img:]
  }

  # get the automatically labeld annotations

  return dataset


def get_image_array(image_input, width, height, imgNorm="sub_mean",
                  ordering='channels_first'):
    """
    Load image array from input
    """

    img = image_input

    if imgNorm == "sub_and_divide":
        img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
    elif imgNorm == "sub_mean":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68
        img = img[:, :, ::-1]
    elif imgNorm == "divide":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img = img/255.0

    if ordering == 'channels_first':
        img = np.rollaxis(img, 2, 0)
    return img

def get_dataset(dataset_folder, kaggle_api_token_path=None):
  """
  download dataset with fitting Kaggle Key
  """
  if os.path.isdir(dataset_folder) and len(os.listdir(dataset_folder)) == 0:
    print("The dataset folder is not empty, not downloading the dataset")
    return

  with open(kaggle_api_token_path) as json_file:
    kaggle_json = json.loads(json_file.read())
    os.environ['KAGGLE_USERNAME'] = kaggle_json["username"]
    os.environ['KAGGLE_KEY'] = kaggle_json["key"]

  import kaggle

  kaggle.api.authenticate()
  os.environ['KAGGLE_USERNAME'] = ""
  os.environ['KAGGLE_KEY'] = ""
  kaggle.api.dataset_download_files('pietbroemmel/naodevils-segmentation-upper-camera', path=dataset_folder, unzip=True)
 

