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