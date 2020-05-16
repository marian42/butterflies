import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import numpy as np
from skimage import io, transform
import csv
import random
import math
from config import *

WHITE_THRESHOLD = 0.95

def clip_image(image):
    coords = ((image[0, :, :] < WHITE_THRESHOLD) | (image[1, :, :] < WHITE_THRESHOLD) | (image[2, :, :] < WHITE_THRESHOLD)).nonzero()
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
        file = open(ROTATION_DATA_FILENAME, 'r')
        reader = csv.reader(file)
        
        self.image_ids, self.angles = zip(*tuple(reader))
        self.angles = list(float(angle) for angle in self.angles)
        
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        file_name = ('data/images_128/{:s}.jpg').format(self.image_ids[index])
        image = io.imread(file_name)

        angle = random.random() * 10 - 5 + random.randint(0, 3) * 90
        image = transform.rotate(image, -self.angles[index] + angle, resize=True, clip=True, mode='constant', cval=1)
        image = torch.tensor(image.transpose((2, 0, 1)), dtype=torch.float32)
        image = clip_image(image)
        image = F.adaptive_avg_pool2d(image, (ROTATION_NETWORK_RESOLUTION, ROTATION_NETWORK_RESOLUTION))

        return image, torch.tensor((math.sin(math.radians(angle)), math.cos(math.radians(angle))))