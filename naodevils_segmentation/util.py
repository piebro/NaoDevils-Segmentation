from imgaug import augmenters as iaa

ALL_AUGMENTATIONS = {}


def add_augmentation(name, augmentation):
  '''
  add augmentation to ALL_AUGMENTATIONS
  '''
  ALL_AUGMENTATIONS[name] = augmentation


aug_low = iaa.Sequential([
  iaa.Fliplr(0.5),

  iaa.Sometimes(0.2, iaa.CropAndPad(
    percent=(-0.1, 0.1),
  )),

  iaa.Sometimes(0.2, iaa.Affine(
    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
    rotate=(-45, 45),
    shear=(-16, 16),
    order=[0, 1]
  ))
  ],
  random_order=True)


aug_mid = iaa.Sequential([
  iaa.Fliplr(0.5),
  
  iaa.Sometimes(0.2, iaa.CropAndPad(
    percent=(-0.1, 0.1),
  )),

  iaa.Sometimes(0.2, iaa.Affine(
    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
    rotate=(-45, 45),
    shear=(-16, 16),
    order=[0, 1]
  )),

  iaa.SomeOf((0, 3),[
    iaa.MotionBlur(k=15),
    iaa.Add((-10, 10), per_channel=0.5),
    iaa.AddToHueAndSaturation((-20, 20)),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255),per_channel=0.5),
    iaa.Multiply((0.5, 1.5), per_channel=0.5)
  ])
  ],
  random_order=True)


aug_high = iaa.Sequential([
  iaa.Fliplr(0.5),
  
  iaa.Sometimes(0.2, iaa.CropAndPad(
    percent=(-0.1, 0.1),
  )),

  iaa.Sometimes(0.2, iaa.Affine(
    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
    rotate=(-45, 45),
    shear=(-16, 16),
    order=[0, 1]
  )),

  iaa.SomeOf((0, 5),[
    iaa.MotionBlur(k=25),
    iaa.Add((-10, 10), per_channel=0.5),
    iaa.AddToHueAndSaturation((-20, 20)),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255),per_channel=0.5),
    iaa.Multiply((0.5, 1.5), per_channel=0.5),
    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
    iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
  ])
  ],
  random_order=True)


add_augmentation("none", None)
add_augmentation("aug_low", aug_low)
add_augmentation("aug_mid", aug_mid)
add_augmentation("aug_high", aug_high)