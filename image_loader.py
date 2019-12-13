import torch
from torch.utils.data import Dataset
import os
import numpy as np
import glob
from skimage import io, transform


def load_image(file_name, resolution=128):
    image = io.imread(file_name)

    image = transform.resize(image, (resolution, resolution), preserve_range=True)    
    image = image.transpose((2, 0, 1)).astype(np.float32) / 255

    return torch.from_numpy(image)

class ImageDataset(Dataset):
    def __init__(self, resolution=128):
        file_names = glob.glob(os.path.join('data/images/', '**.jpg'), recursive=True)
        self.hashes = [f.split('/')[-1][:-4] for f in file_names]
        self.resolution = resolution
        
    def __len__(self):
        return len(self.hashes)

    def __getitem__(self, index):
        hash = self.hashes[index]
        image_file_name = 'data/images/{:s}.jpg'.format(hash)
        image = load_image(image_file_name)
        
        image = io.imread(image_file_name)

        image = transform.resize(image, (self.resolution, self.resolution), preserve_range=True)    
        image = image.transpose((2, 0, 1)).astype(np.float32) / 255

        return torch.from_numpy(image)

