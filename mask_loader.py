import torch
from torch.utils.data import Dataset
import os
import numpy as np
import glob
from skimage import io

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

        mask = io.imread(mask_file_name)
        mask = mask[:, :, 0].astype(np.float32) / 255
        width = mask.shape[0] // 16 * 16
        height = mask.shape[1] // 16 * 16
        mask = mask[:width, :height]
        mask = torch.from_numpy(mask)


        image = io.imread(image_file_name)
        image = image.transpose((2, 0, 1)).astype(np.float32) / 255
        image = image[:, :width, :height]
        image = torch.from_numpy(image)

        self.cache[index] = (image, mask, hash)
        return image, mask, hash