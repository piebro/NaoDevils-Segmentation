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

def draw_annotation_segmentation(annotation, height=480, width=640):
  #anns = coco.loadAnns(ann_ids)    
  #img_infos = coco.loadImgs(img_id)[0] 
  mask = np.zeros((height, width))

  mask_list = []
  mask_list_rb = []
  robot_count = 0
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
    if category_id<3:
      mask[_mask == 1] = category_id
    elif category_id == 3: # categorie is robot
      mask[_mask == 1] = 6 + robot_count
      robot_count += 1
    else:
      mask[_mask == 1] = category_id-1

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