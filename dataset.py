import cv2
from os.path import join

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize

from albumentations import ShiftScaleRotate, HorizontalFlip
from albumentations import Compose as Compose_alb

img_dir = './data/train/images'
mask_dir = './data/train/masks'


def aug():
    return Compose_alb([
        HorizontalFlip(0.5)
    ])


input_normalizer = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225])
])


class SaltTrainDataset(Dataset):
    def __init__(self, fold, input_size, is_train=True):
        self.is_train = is_train
        self.img_pathes = fold
        self.input_size = input_size

    def __len__(self):
        return len(self.img_pathes)

    def __getitem__(self, id):
        image = cv2.imread(join(img_dir, self.img_pathes[id]))[:,:,::-1]
        mask = cv2.imread(join(mask_dir, self.img_pathes[id]), 0) / 255
        if self.is_train:
            augmentation = aug()
            data = {'image': image, 'mask': mask}
            augmented = augmentation(**data)
            image, mask = augmented['image'], augmented['mask']
        image = cv2.resize(image, (self.input_size, self.input_size))
        mask = cv2.resize(mask, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
        return input_normalizer(image), torch.from_numpy(mask).float().unsqueeze_(0)


test_img_dir = './data/test/images'


class SaltInferenceDataset(Dataset):
    def __init__(self, img_pathes, input_size):
        self.img_pathes = img_pathes
        self.input_size = input_size

    def __len__(self):
        return len(self.img_pathes)

    def __getitem__(self, id):
        image = cv2.imread(join(test_img_dir, self.img_pathes[id]))[:,:,::-1]
        image = cv2.resize(image, (self.input_size, self.input_size))
        meta = {
            'img_name': self.img_pathes[id]
        }
        return input_normalizer(image), meta