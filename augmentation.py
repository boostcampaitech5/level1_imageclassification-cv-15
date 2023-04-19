from albumentations.pytorch.transforms import ToTensorV2
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, ToGray, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize, Rotate
)
# from torchvision.transforms import Resize, ToTensor, Normalize, Compose, CenterCrop, ColorJitter, Grayscale, RandomHorizontalFlip
from PIL import Image
import torch
import cv2

class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = Compose([
            Resize(*resize),
            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_REPLICATE),
            ToTensorV2(),
        ])

    def __call__(self, image):
        return self.transform(image=image)
    
class GrayAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = Compose([
            Resize(*resize),
            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToGray(True),
            ToTensorV2(),
        ])

    def __call__(self, image):
        return self.transform(image=image)

class AddGaussianNoise(object):
    """
        transform 에 없는 기능들은 이런식으로 __init__, __call__, __repr__ 부분을
        직접 구현하여 사용할 수 있습니다.
    """

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CustomAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = Compose([
            CenterCrop(320, 256),
            Resize(*resize),
            ToTensorV2(),
            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            AddGaussianNoise()
        ])

    def __call__(self, image):
        return self.transform(image=image)
    
class MyAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = Compose([
            CenterCrop(320, 256),
            Resize(*resize),
            ToTensorV2(),
            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            HorizontalFlip()
        ])

    def __call__(self, image):
        return self.transform(image=image)
class AugV2:
    def __init__(self, resize, mean, std, **args):
        self.transform = Compose([
                  Transpose(p=0.5),
                  HorizontalFlip(p=0.5),
                  VerticalFlip(p=0.5),
                  ShiftScaleRotate(p=0.5),
                  RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                  Resize(*resize),
                  Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                  ToTensorV2(p=1.0),
                  ], p=1.)
    def __call__(self, image):
        return self.transform(image=image)