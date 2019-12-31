import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import numpy as np
import glob
from skimage import io, transform
import csv
import random
import math

RESOLUTION = 64
USE_ALPHA_IMAGES = False

def clip_image(image):
    coords = ((image[0, :, :] < 1) | (image[1, :, :] < 1) | (image[2, :, :] < 1)).nonzero()
    top_left, _ = torch.min(coords, dim=0)
    bottom_right, _ = torch.max(coords, dim=0)
    image = image[:, top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

    new_size = max(image.shape[1], image.shape[2])
    result = torch.zeros((3, new_size, new_size), dtype=torch.float32)
    y, x = (new_size - image.shape[1]) // 2, (new_size - image.shape[2]) // 2
    result[:, :, :] = 1
    result[:, y:y+image.shape[1], x:x+image.shape[2]] = image
    return result

class RotationDataset(Dataset):
    def __init__(self, return_hashes=False):
        file = open('data/rotations.csv', 'r')
        reader = csv.reader(file)
        
        self.image_ids, self.angles = zip(*tuple(reader))
        self.angles = list(float(angle) for angle in self.angles)
        
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        file_name = ('data/images_alpha/{:s}.png' if USE_ALPHA_IMAGES else 'data/images_128/{:s}.jpg').format(self.image_ids[index])
        image = io.imread(file_name)

        angle = random.randrange(0, 360)
        image = transform.rotate(image[:, :, :3] if USE_ALPHA_IMAGES else image, -self.angles[index] + angle, resize=True, clip=True, mode='constant', cval=1)
        image = torch.from_numpy(image.transpose((2, 0, 1)))
        image = clip_image(image)
        image = F.adaptive_avg_pool2d(image, (RESOLUTION, RESOLUTION))

        return image, torch.tensor((math.sin(math.radians(angle)), math.cos(math.radians(angle))))