import torch
from torch.utils.data import Dataset
import numpy as np
import glob
from skimage import io
import csv
from config import QUALITY_CALCULATED_FILENAME

class ImageDataset(Dataset):
    def __init__(self, return_hashes=False, quality=None):
        if quality is None:
            file_names = glob.glob('data/images_rotated_128/**.jpg', recursive=True)
            self.hashes = sorted([f.split('/')[-1][:-4] for f in file_names])
        else:
            file = open(QUALITY_CALCULATED_FILENAME, 'r')
            reader = csv.reader(file)

            if isinstance(quality, int):
                self.hashes = [row[0] for row in reader if int(row[1]) == quality]
            elif isinstance(quality, tuple) or isinstance(quality, list):
                self.hashes = [row[0] for row in reader if int(row[1]) in quality]
        
        self.return_hashes = return_hashes
        
    def __len__(self):
        return len(self.hashes)

    def __getitem__(self, index):
        hash = self.hashes[index]
        image_file_name = 'data/images_rotated_128/{:s}.jpg'.format(hash)
        
        image = io.imread(image_file_name)

        image = image.transpose((2, 0, 1)).astype(np.float32) / 255
        image = torch.from_numpy(image)

        if self.return_hashes:
            return image, hash
        else:
            return image
