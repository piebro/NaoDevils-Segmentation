import os
import glob

import naodevils_segmentation.dataset_loader as dataset_loader

import numpy as np
import imgaug as ia
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import itertools

ALL_MODELS = {}

ALL_AUGMENTATIONS = {}

ALL_GET_MASK_FUNCTIONS = {}

def add_model(name, get_model_function):
  '''
  add model to ALL_MODELS
  '''
  ALL_MODELS[name] = get_model_function

def add_augmentation(name, augmentation):
  '''
  add augmentation to ALL_AUGMENTATIONS
  '''
  ALL_AUGMENTATIONS[name] = augmentation

def add_get_mask_function(name, get_mask_function, n_classes):
  '''
  add get_mask_function to ALL_GET_MASK_FUNCTIONS
  '''
  ALL_GET_MASK_FUNCTIONS[name] = {
    "func": get_mask_function,
    "n_classes": n_classes   
  }

def get_mask(annotation, height=480, width=640):
  '''
  returns mask with all annotations drawn on the mask
  '''
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


def show_images(data_show, get_mask_function, num_of_images=4, augmentation=None):
  '''
  shows random images as a figure with pyplot from data_show with the given the mask
  '''
  #TODO change shuffeling the data to picking random samples without picking the same ones in the data
  np.random.shuffle(data_show)

  if type(augmentation) == str:
    augmentation = ALL_AUGMENTATIONS[augmentation]

  if type(get_mask_function) == str:
    get_mask_function = ALL_GET_MASK_FUNCTIONS[get_mask_function]["func"]

  for i in range(0, num_of_images, 2):
    img1 = cv2.imread(data_show[i]["img_path"])
    gt1 = get_mask_function(data_show[i]["annotation"])
    img1, gt1 = get_colored_segmentation_mask(img1, gt1, augmentation=augmentation)

    img2 = cv2.imread(data_show[i+1]["img_path"])
    gt2 = get_mask_function(data_show[i+1]["annotation"])
    img2, gt2 = get_colored_segmentation_mask(img2, gt2, augmentation=augmentation)

    fig = plt.figure(figsize=(20,30))

    ax = fig.add_subplot(1,4,1)
    ax.set_axis_off()
    ax.title.set_text("Image id: " + os.path.basename(data_show[i]["img_path"])[:5])
    ax.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))

    ax = fig.add_subplot(1,4,2)
    ax.set_axis_off()
    ax.title.set_text("Groundtruth")
    ax.imshow(gt1)

    ax = fig.add_subplot(1,4,3)
    ax.set_axis_off()
    ax.title.set_text("Image id: " + os.path.basename(data_show[i+1]["img_path"])[:5])
    ax.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    ax = fig.add_subplot(1,4,4)
    ax.set_axis_off()
    ax.title.set_text("Groundtruth")
    ax.imshow(gt2)


def show_prediction(model, data_pred, num_of_images=2, indices=None):
  '''
  shows predicted images with the given model from random samples 
  '''
  #TODO show random images in the range without showing the same one twice
  if indices == None:
    indices = range(num_of_images)

  for i in indices:
    if i >= len(data_pred):
        raise IndexError("Try an index <= {}".format(len(data_pred)))

    data = data_pred[i]
    img = cv2.imread(data["img_path"])


    gt = model.get_mask_function(data["annotation"])
    img, gt = get_colored_segmentation_mask(img, gt)


    img_array = dataset_loader.get_image_array(
        img, model.input_width, model.input_height, ordering="channels_last")
    pred = model.predict(np.array([img_array]))
    pred = np.argmax(pred[0], axis=-1)
    img, pred = get_colored_segmentation_mask(img, pred)


    fig = plt.figure(figsize=(20,30))
    ax = fig.add_subplot(1,3,1)
    ax.set_axis_off()
    image_name_id = os.path.basename(data["img_path"])[:5]
    ax.title.set_text("Image id: " + image_name_id)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    ax = fig.add_subplot(1,3,2)
    ax.set_axis_off()
    ax.title.set_text("Groundtruth")
    ax.imshow(gt)

    ax = fig.add_subplot(1,3,3)
    ax.set_axis_off()
    ax.title.set_text("Prediction")
    ax.imshow(pred)
    fig.show()


def get_colored_segmentation_mask(img, mask, class_colors=None, augmentation=None):
  '''
  color the mask and output it as an rbg image
  '''
  unique = np.unique(mask)

  if class_colors == None:
    cmap = plt.get_cmap("tab10")
    class_colors = np.asarray([cmap(i)[:-1] for i in np.linspace(0, 1, num=10)])*255
    class_colors = class_colors.astype(np.uint8)
  
  seg_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

  if augmentation != None:
    img, mask[:, :] = augment_segmentation(img, mask[:, :], augmentation)

  for c in unique:
    seg_img[mask == c] = class_colors[c]

  return img, seg_img


def get_segmentation_array(image_input, nClasses, width, height, no_reshape=False):
  '''
  Load segmentation array from input
  '''

  seg_labels = np.zeros((height, width, nClasses))

  img = image_input

  img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)

  for c in range(nClasses):
    seg_labels[:, :, c] = (img == c).astype(int)

  if not no_reshape:
    seg_labels = np.reshape(seg_labels, (width*height, nClasses))

  return seg_labels


def get_model_from_str(log_dir, model_str, epoch=None, try_loading_weights=True):
  """
  use a model_str to load a model and fitting weights in the log_dir if wanted
  """
  model_str_list = model_str.split("-")

  input_size_str = model_str_list[1]
  input_height = int(input_size_str.split("x")[0])
  input_width = int(input_size_str.split("x")[1])

  get_mask_function = ALL_GET_MASK_FUNCTIONS[model_str_list[2]]["func"]
  n_classes = ALL_GET_MASK_FUNCTIONS[model_str_list[2]]["n_classes"]

  model = ALL_MODELS[model_str_list[0]](get_mask_function, n_classes, input_height, input_width)
  
  if try_loading_weights:
    cp_dir = os.path.join(log_dir, model_str)
    load_weight(model, cp_dir, model_str, epoch=epoch)
  return model

def load_weight(model, cp_dir, train_str, epoch=None):
  if epoch is None:
    latest_checkpoint = sorted(glob.glob(os.path.join(cp_dir, train_str + "_weights.*")))[-1]
  else:
    latest_checkpoint = os.path.join(cp_dir, train_str + "_weights.{:03d}.hdf5".format(epoch))

  model.load_weights(latest_checkpoint)
  print("loaded weights ", latest_checkpoint)
  return int(latest_checkpoint[-8:-5])


def train_with_str(log_dir, 
                   data_train,
                   data_val,
                   model_str,
                   epochs,
                   batch_size=4,
                   optimizer_name='adadelta',
                   metrics=['accuracy'],
                   loss='categorical_crossentropy',
                   steps_per_epoch=None,
                   validation_steps=None
                   ):
  """
  train with a model string and save all weight in the log_dir
  """                   
  model = get_model_from_str(log_dir, model_str, try_loading_weights=False)

  print('\nParameter Count:', model.count_params())

  aug_str = model_str.split("-")[3]
  augmentation = ALL_AUGMENTATIONS[aug_str]
  if augmentation==None:
    print("no augmentation")

  train(model,
      data_train,
      data_val,
      log_dir,
      train_str=model_str,
      epochs=epochs,
      batch_size=batch_size,
      optimizer_name=optimizer_name,
      augmentation=augmentation,
      metrics=metrics,
      loss=loss,
      steps_per_epoch=steps_per_epoch,
      validation_steps=validation_steps
      )

def train(model,
          data_train,
          data_val,
          log_dir,
          train_str="test",
          epochs=5,
          batch_size=4,
          optimizer_name='adadelta',
          augmentation=None,
          metrics=['accuracy'],
          loss='categorical_crossentropy',
          steps_per_epoch=None,
          validation_steps=None
          ):
  """
  train given model and save weights in the log_dir
  """

  train_gen = image_segmentation_generator(
    data_train, batch_size, model, augmentation=augmentation)


  val_gen = image_segmentation_generator(
    data_val, batch_size, model)
  
  if steps_per_epoch == None:
    steps_per_epoch = int(len(data_train)/batch_size)
  if validation_steps == None:
    validation_steps = int(len(data_val)/batch_size*0.8)

  cp_dir = os.path.join(log_dir,train_str)
  print("Model dir is: " + cp_dir)
  if os.path.isdir(cp_dir):
    start_epoch = load_weight(model, cp_dir, train_str)
  else:
    os.mkdir(cp_dir)
    start_epoch = 0
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=cp_dir)

  save_model_path = os.path.join(cp_dir, train_str + "_weights.{epoch:03d}.hdf5")
  save_callback = tf.keras.callbacks.ModelCheckpoint(save_model_path)

  model.compile(loss=loss,
                optimizer=optimizer_name,
                metrics=metrics)

  model.fit_generator(train_gen,
            steps_per_epoch,
            validation_data=val_gen,
            validation_steps=validation_steps,
            epochs=epochs,
            callbacks=[tensorboard_callback, save_callback],
            initial_epoch=start_epoch)    
  return model


def image_segmentation_generator(data_list, batch_size, model, augmentation=None):
  """
  generator for the sagmentation models
  """
  n_classes = model.n_classes
  input_height = model.input_height
  input_width = model.input_width
  output_height = model.output_height
  output_width = model.output_width

  zipped = itertools.cycle(data_list)

  while True:
    X = []
    Y = []
    for _ in range(batch_size):
      data = next(zipped)

      img = cv2.imread(data["img_path"])
      mask = model.get_mask_function(data["annotation"])

      if augmentation != None:
        img, mask = augment_segmentation(img, mask , augmentation=augmentation)

      X.append(dataset_loader.get_image_array(
        img, input_width, input_height, ordering="channels_last"))

      Y.append(get_segmentation_array(
        mask, n_classes, output_width, output_height, no_reshape=True))

      # todo: wenn meta info gegebe, dies auch ausgeben, oder das als parameter entscheiden

    # [None], because of: https://stackoverflow.com/a/60131716
    yield np.array(X), np.array(Y), [None]


def augment_segmentation(img, seg , augmentation):
  """
  augment given image and segmentation mask with an imaug augmentation
  """
  # Create a deterministic augmentation from the random one
  aug_det = augmentation.to_deterministic()
  # Augment the input image
  image_aug = aug_det.augment_image(img)

  segmap = ia.SegmentationMapOnImage(seg, nb_classes=np.max(seg) + 1, shape=img.shape)
  segmap_aug = aug_det.augment_segmentation_maps(segmap)
  segmap_aug = segmap_aug.get_arr_int()

  return image_aug, segmap_aug
