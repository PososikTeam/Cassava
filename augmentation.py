import albumentations as A
from albumentations.pytorch import ToTensor
from torch.utils.data import Dataset, DataLoader
import cv2

BORDER_CONSTANT = 0
BORDER_REFLECT = 2
# image_size = (224, 224)

# def pre_transforms(image_size = image_size[0]):
#     # Convert the image to a square of size image_size x image_size
#     # (keeping aspect ratio)
#     result = [
#         A.LongestMaxSize(max_size=image_size),
#     ]
    
#     return result

img_min_size = 600

def get_super_light_augmentations(image_size):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ])

def get_light_augmentations(image_size):
    min_size = min(image_size[0], image_size[1])
    return A.Compose([
        A.RandomSizedCrop(min_max_height=(int(min_size * 0.85), min_size),
                          height=image_size[0],
                          width=image_size[1], p=1.0),

        A.OneOf([A.HorizontalFlip(),
                A.VerticalFlip(),
                A.RandomRotate90()
                ])  
    ])

def get_kaggle_light_augmentations(image_size):
    min_size = min(image_size[0], image_size[1])
    return A.Compose([
        A.RandomSizedCrop(min_max_height=(int(min_size * 0.85), min_size),
                          height=image_size[0],
                          width=image_size[1], p=1.0),

        A.OneOf([A.HorizontalFlip(),
                A.VerticalFlip(),
                A.RandomRotate90(),
                A.NoOp()
                ]),  
        A.Cutout(p=0.5, num_holes=8, max_h_size=16, max_w_size=16)
    ])

def get_medium_augmentations(image_size):
    min_size = min(image_size[0], image_size[1])

    return A.Compose([
        A.OneOf([A.RandomSizedCrop(min_max_height=(int(min_size* 0.85), min_size),
                          height=image_size[0],
                          width=image_size[1]),
                A.Resize(image_size[0], image_size[1]),
                A.CenterCrop(image_size[0], image_size[1])
                ], p = 1.0),

        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.15,
                                       contrast_limit=0.5),
            A.RandomGamma(gamma_limit=(50, 150)),
            A.NoOp()
        ], p = 1.0),
        
        A.OneOf([A.CLAHE(p=0.5, clip_limit=(10, 10), tile_grid_size=(3, 3)),
                A.FancyPCA(alpha=0.4),
                A.NoOp(),
                ], p = 1.0),
        
        A.OneOf([A.HorizontalFlip(),
                A.VerticalFlip(),
                A.RandomRotate90(),
                A.NoOp()
                ], p = 1.0)
    ])

def get_hard_augmentations(image_size):
    return None

# def get_medium_augmentations(image_size):
#     return A.Compose([
#         A.OneOf([
#             A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
#                                rotate_limit=15,
#                                border_mode=cv2.BORDER_CONSTANT, value=0),
#             A.OpticalDistortion(distort_limit=0.11, shift_limit=0.15,
#                                 border_mode=cv2.BORDER_CONSTANT,
#                                 value=0),
#             A.NoOp()
#         ]),
#         A.RandomSizedCrop(min_max_height=(int(image_size[0] * 0.75), image_size[0]),
#                           height=image_size[0],
#                           width=image_size[1], p=0.3),
#         A.OneOf([
#             A.RandomBrightnessContrast(brightness_limit=0.5,
#                                        contrast_limit=0.4),
#             A.RandomGamma(gamma_limit=(50, 150)),
#             A.NoOp()
#         ]),
#         A.OneOf([
#             A.RGBShift(r_shift_limit=20, b_shift_limit=15, g_shift_limit=15),
#             A.HueSaturationValue(hue_shift_limit=5,
#                                  sat_shift_limit=5),
#             A.NoOp()
#         ]),
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5)
#     ])

def post_transforms():
    # we use ImageNet image normalization
    # and convert it to torch.Tensor
#     return [ToTensor()]
    return A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406),
                       std=(0.229, 0.224, 0.225)), ToTensor()])

# def compose(transforms_to_compose):
#     # combine all augmentations into one single pipeline
#     result = A.Compose([
#       item for sublist in transforms_to_compose for item in sublist
#     ])
#     return result


def get_test_transform(image_size):
    return A.Compose([A.Resize(image_size[0], image_size[1], p = 1.0), A.Normalize(mean=(0.485, 0.456, 0.406),
                       std=(0.229, 0.224, 0.225), p = 1.0), ToTensor()])

def get_train_transform(augmentation, image_size):
    LEVELS = {
        'super_light': get_super_light_augmentations,
        'kaggle_light': get_kaggle_light_augmentations,
        'light': get_light_augmentations,
        'medium': get_medium_augmentations,
        'hard': get_hard_augmentations,
#         'hard2': get_hard_augmentations_v2
    }

    aug = LEVELS[augmentation](image_size)

    return A.Compose([aug, post_transforms()])
    
    
