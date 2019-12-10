from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import sys
from torch.utils.data import DataLoader
from torchvision import utils
import random
import glob
from skimage import io
import os
from shutil import copyfile
from mask_loader import load_image

from classifier import Classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSIFIER_FILENAME = 'trained_models/classifier.to'

classifier = Classifier()
classifier.cuda()
classifier.load_state_dict(torch.load(CLASSIFIER_FILENAME))
classifier.eval()

file_names = glob.glob('data/raw/**.jpg', recursive=True)

def clip_mask(mask, clipping_range = 0.2):
    mask = (mask - 0.5) * 2
    mask.clamp_(-clipping_range, clipping_range)
    mask /= clipping_range
    mask = mask / 2 + 0.5
    return mask

with torch.no_grad():
    while True:
        file_name = random.choice(file_names)
        hash = file_name.split('/')[-1][:-4]

        image = load_image(file_name).to(device)
        mask = classifier(image.unsqueeze(0)).squeeze(0)
        mask = clip_mask(mask)
        
        
        copyfile(file_name, 'data/test/{:s}.jpg'.format(hash))
        utils.save_image(mask, 'data/test/{:s}_mask.png'.format(hash))
        image *= mask
        utils.save_image(image, 'data/test/{:s}_result.png'.format(hash))