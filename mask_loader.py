import torch
from torch.utils.data import Dataset
import os
import numpy as np
import glob
from skimage import io, transform

def load_image(file_name, is_bw=False):
    image = io.imread(file_name)

    while min(image.shape[0], image.shape[1]) >= 1024:
        image = transform.resize(image, (image.shape[0] // 2, image.shape[1] // 2), preserve_range=True)
    
    width = image.shape[0] // 16 * 16
    height = image.shape[1] // 16 * 16
    image = image.transpose((2, 0, 1)).astype(np.float32) / 255
    image = image[:, :width, :height]

    if is_bw:
        image = image[0, :, :]

    return torch.from_numpy(image)

class MaskDataset(Dataset):
    def __init__(self):
        file_names = glob.glob(os.path.join('data/masks/', '**.png'), recursive=True)
        self.hashes = [f.split('/')[-1][:-4] for f in file_names]
        self.cache = dict()
        
    def __len__(self):
        return len(self.hashes)

    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]

        hash = self.hashes[index]
        mask_file_name = 'data/masks/{:s}.png'.format(hash)
        image_file_name = 'data/raw/{:s}.jpg'.format(hash)

        mask = load_image(mask_file_name, is_bw=True)
        image = load_image(image_file_name)

        self.cache[index] = (image, mask, hash)
        return image, mask, hash

