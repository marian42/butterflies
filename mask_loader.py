import torch
from torch.utils.data import Dataset
import numpy as np
import glob
from skimage import io, transform
import random

WIDTH = 1152 # 128 * 9
HEIGHT = 768 # 128 * 6

def load_image(file_name, is_bw=False, rotation = None):
    image = io.imread(file_name)

    if rotation is not None:
        image = np.rot90(image, k=rotation)

    while image.shape[0] > HEIGHT or image.shape[1] > WIDTH:
        image = transform.resize(image, (image.shape[0] // 2, image.shape[1] // 2), preserve_range=True)

    if is_bw:
        result = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        result[:image.shape[0], :image.shape[1]] = image[:, :, 0].astype(np.float32) / 255
    else:
        image = image.transpose((2, 0, 1)).astype(np.float32) / 255
        result = np.ones((3, HEIGHT, WIDTH), dtype=np.float32)
        result[:, :image.shape[1], :image.shape[2]] = image

    return torch.from_numpy(result)

class MaskDataset(Dataset):
    def __init__(self):
        file_names = glob.glob('data/masks/**.png', recursive=True)
        self.hashes = [f.split('/')[-1][:-4] for f in file_names]
        
    def __len__(self):
        return len(self.hashes)

    def __getitem__(self, index):
        hash = self.hashes[index]
        mask_file_name = 'data/masks/{:s}.png'.format(hash)
        image_file_name = 'data/raw/{:s}.jpg'.format(hash)

        rotation = random.randint(0, 3)

        mask = load_image(mask_file_name, is_bw=True, rotation=rotation)
        image = load_image(image_file_name, rotation=rotation)
        
        return image, mask, hash

