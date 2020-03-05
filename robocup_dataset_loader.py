import sys
import io
import os
import json

import png
import imageLabelData_pb2
import itertools
import cv2
from pycocotools.coco import COCO
import numpy as np
from google.protobuf.json_format import MessageToJson
import imgaug as ia
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt


def get_data_list(dataset_folder, dataset_type=None, dataset_validation_size=0.2, ran_seed=42, get_meta_info=False):
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
      img_file_name = coco.loadImgs(img_id)[0]['file_name']
      img_path = os.path.join(folder_name, img_file_name)
      img_path = os.path.join(dataset_folder, img_path)

      data_entry = {
        "img_path": img_path,
        "annotation": coco.loadAnns(coco.getAnnIds(img_id))
      }
      
      if get_meta_info:
        meta_info = get_image_meta_infos(img_path)
        data_entry["meta_info"] = meta_info
      
      img_paths_annotations.append(data_entry)
  
  np.random.seed(ran_seed)
  np.random.shuffle(img_paths_annotations)
  num_of_val_img = round(len(img_paths_annotations)*dataset_validation_size)
  if(dataset_type=="training"):
    img_paths_annotations = img_paths_annotations[:-num_of_val_img]
  if(dataset_type=="validation"):
    img_paths_annotations = img_paths_annotations[-num_of_val_img:]
  
  return img_paths_annotations

def draw_annotation_segmentation(annotation, height=480, width=640):
  mask = np.zeros((height, width), dtype=np.uint8)

  mask_list = []
  mask_list_rb = []
  for ann in annotation:
    segmentations = ann['segmentation']
    category_id = ann['category_id']
    pts = [
          np
          .array(anno)
          .reshape(-1, 2)
          .round()
          .astype(int)
          for anno in segmentations
          ]
        
    _mask = mask.copy()
    cv2.fillPoly(_mask, pts, 1)

    if category_id == 2 or category_id == 3: # ball or robot
      mask_list_rb.append((category_id, _mask))
    else:
      mask_list.append((category_id, _mask))

  mask_list = sorted(mask_list, key=lambda x: x[0])
  mask_list.extend(mask_list_rb)

  for category_id, _mask in mask_list:
    mask[_mask == 1] = category_id

  return mask

def draw_annotation_mask_rcnn(annotation, height=480, width=640):
  count = len(annotation)
  mask = np.zeros([height, width, count], dtype=np.uint8)
  
  seg_list = []
  seg_list_rb = []
  for shape in annotation:
    category_id = shape["category_id"]
    segmentations  = shape["segmentation"]
    if category_id == 2 or category_id == 3:
        seg_list_rb.append((category_id, segmentations))
    else:
        seg_list.append((category_id, segmentations))
  
  seg_list = sorted(seg_list, key=lambda x: x[0])
  seg_list.extend(seg_list_rb)
  
  category_id_list = []
  for i, (category_id, segmentations) in enumerate(seg_list):
    category_id_list.append(category_id)

    pts = [
      np
      .array(anno)
      .reshape(-1, 2)
      .round()
      .astype(int)
      for anno in segmentations
      ]
        
    img = mask[:, :, i:i+1].copy()
    cv2.fillPoly(img, pts, 1)
    mask[:, :, i:i+1] = img
    
  # Handle occlusions
  if(mask.shape[2] > 0): # if at least one mask is there
    occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
    for i in range(count-2, -1, -1):
        mask[:, :, i] = mask[:, :, i] * occlusion
        occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))

  return mask.astype(np.bool), np.array(category_id_list).astype(np.int32)


def get_image_meta_infos(img_path):
  protobuf_chunk = read_label_chunk(img_path)
  data = imageLabelData_pb2.ImageLabelData()
  data.ParseFromString(protobuf_chunk)
  data_json = MessageToJson(data)
  return json.loads(data_json)

def read_label_chunk(img_path):
    p = png.Reader(img_path)
    for chunk_name, chunk_data in p.chunks():
        if chunk_name == b'laBl':
            return chunk_data

def get_colored_segmentation_mask(img, mask, class_colors=None, augmentation=None):
  unique = np.unique(mask)

  if class_colors == None:
    cmap = plt.get_cmap("tab10")
    class_colors = np.asarray([cmap(i)[:-1] for i in np.linspace(0, 1, num=10)])*255
    class_colors = class_colors.astype(np.uint8)
  
  seg_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

  if augmentation != None:
      img, mask[:, :] = augment_seg(img, mask[:, :], augmentation)

  for c in unique:
    seg_img[mask == c] = class_colors[c]

  return img, seg_img

def augment_seg(img, seg , augmentation):
    # Create a deterministic augmentation from the random one
    aug_det = augmentation.to_deterministic()
    # Augment the input image
    image_aug = aug_det.augment_image(img)

    segmap = ia.SegmentationMapOnImage(seg, nb_classes=np.max(seg) + 1, shape=img.shape)
    segmap_aug = aug_det.augment_segmentation_maps(segmap)
    segmap_aug = segmap_aug.get_arr_int()

    return image_aug, segmap_aug

def image_segmentation_generator(data_list, batch_size,
                                 n_classes, input_height, input_width,
                                 output_height, output_width, augmentation=None,
                                 draw_annotation_func = None):
  
    zipped = itertools.cycle(data_list)

    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            data = next(zipped)

            img = cv2.imread(data["img_path"], 1)
            if draw_annotation_func == None:
              mask = draw_annotation_segmentation(data["annotation"])
            else:
              mask = draw_annotation_func(data["annotation"])
            
            if augmentation != None:
                img, mask = augment_seg(img, mask , augmentation=augmentation )

            X.append(get_image_array(img, input_width,
                                   input_height, ordering="channels_last"))
            
            Y.append(get_segmentation_array(
                mask, n_classes, output_width, output_height))

        yield np.array(X), np.array(Y)

def get_segmentation_array(image_input, nClasses, width, height, no_reshape=False):
    """ Load segmentation array from input """

    seg_labels = np.zeros((height, width, nClasses))

    img = image_input

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)

    for c in range(nClasses):
        seg_labels[:, :, c] = (img == c).astype(int)

    if not no_reshape:
        seg_labels = np.reshape(seg_labels, (width*height, nClasses))

    return seg_labels


def get_image_array(image_input, width, height, imgNorm="sub_mean",
                  ordering='channels_first'):
    """ Load image array from input """

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
  with open(kaggle_api_token_path) as json_file:
    kaggle_json = json.loads(json_file.read())
    os.environ['KAGGLE_USERNAME'] = kaggle_json["username"]
    os.environ['KAGGLE_KEY'] = kaggle_json["key"]

  import kaggle

  kaggle.api.authenticate()
  os.environ['KAGGLE_USERNAME'] = ""
  os.environ['KAGGLE_KEY'] = ""
  kaggle.api.dataset_download_files('pietbroemmel/naodevils-segmentation-upper-camera', path=dataset_folder, unzip=True)
 

