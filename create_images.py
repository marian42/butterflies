from itertools import count

import torch
from torchvision import utils
import torch.nn.functional as F
import random
import glob
from shutil import copyfile
from mask_loader import load_image
from tqdm import tqdm
import os
from torch.utils.data import DataLoader, Dataset

from classifier import Classifier

OUTPUT_RESOLUTION = None
MARGIN = 0.1
ALPHA = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSIFIER_FILENAME = 'trained_models/classifier.to'

classifier = Classifier()
classifier.cuda()
classifier.load_state_dict(torch.load(CLASSIFIER_FILENAME))
classifier.eval()

SKIP_ITEM = 0

class ImageDataset(Dataset):
    def __init__(self):
        file_names = file_names = glob.glob('data/raw/**.jpg', recursive=True)
        self.hashes = [f.split('/')[-1][:-4] for f in file_names]
        
    def __len__(self):
        return len(self.hashes)

    def __getitem__(self, index):
        hash = self.hashes[index]
        image_file_name = 'data/raw/{:s}.jpg'.format(hash)
        result_file_name = 'data/images' \
            + ('' if OUTPUT_RESOLUTION is None else '_{:d}'.format(OUTPUT_RESOLUTION)) \
            + ('' if not ALPHA else '_alpha') \
            + '/{:s}.{:s}'.format(hash, 'png' if ALPHA else 'jpg')

        if os.path.exists(result_file_name):
            return SKIP_ITEM

        try:
            image = load_image(image_file_name)
        except ValueError:
            print("Could not open {:s}.".format(image_file_name))
            return SKIP_ITEM
        
        return image, result_file_name

dataset = ImageDataset()
data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

for item in tqdm(data_loader):
    if item == SKIP_ITEM:
        continue

    image, result_file_name = item
    image = image.to(device)

    image = classifier.apply(image, margin=MARGIN, create_alpha=ALPHA)
    
    if image is None:
        print("Found nothing.")
        continue
    
    if OUTPUT_RESOLUTION is not None:
        image = (F.adaptive_avg_pool2d(image, (OUTPUT_RESOLUTION, OUTPUT_RESOLUTION))).data
        
    utils.save_image(image, result_file_name[0])