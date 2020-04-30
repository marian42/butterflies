import torch
from torch.utils.data import Dataset
import numpy as np
import glob
from skimage import io, transform
import random

WIDTH = 512
HEIGHT = 512

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

        image = io.imread(image_file_name)
        image = np.rot90(image, k=rotation)
        image = image.transpose((2, 0, 1)).astype(np.float32) / 255

        x_pos = 0 if image.shape[2] <= WIDTH else random.randrange(0, image.shape[2] - WIDTH)
        y_pos = 0 if image.shape[1] <= HEIGHT else random.randrange(0, image.shape[1] - HEIGHT)

        image = image[:, y_pos:y_pos+HEIGHT, x_pos:x_pos+WIDTH]

        if image.shape[1] < HEIGHT or image.shape[2] < WIDTH:
            new_image = np.ones((3, HEIGHT, WIDTH), dtype=np.float32)
            new_image[:, :image.shape[1], :image.shape[2]] = image
            image = torch.from_numpy(new_image)
        else:
            image = torch.from_numpy(image)

        mask = io.imread(mask_file_name)
        mask = np.rot90(mask, k=rotation)

        mask = mask[y_pos:y_pos+HEIGHT, x_pos:x_pos+WIDTH, 0].astype(np.float32) / 255

        if mask.shape[0] < HEIGHT or mask.shape[1] < WIDTH:
            new_mask = np.ones((HEIGHT, WIDTH), dtype=np.float32)
            new_mask[:mask.shape[0], :mask.shape[1]] = mask
            mask = torch.from_numpy(new_mask)
        else:
            mask = torch.from_numpy(mask)
        
        return image, mask, hash

