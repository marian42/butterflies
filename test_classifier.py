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

from classifier import Classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSIFIER_FILENAME = 'trained_models/classifier.to'

classifier = Classifier()
classifier.cuda()
classifier.load_state_dict(torch.load(CLASSIFIER_FILENAME))
classifier.eval()

file_names = glob.glob('data/raw/**.jpg', recursive=True)

with torch.no_grad():
    while True:
        file_name = random.choice(file_names)
        hash = file_name.split('/')[-1][:-4]

        image = io.imread(file_name)
        image = image.transpose((2, 0, 1)).astype(np.float32) / 255
        width = image.shape[1] // 16 * 16
        height = image.shape[2] // 16 * 16
        image = image[:, :width, :height]
        image = torch.from_numpy(image).to(device)
        mask = classifier(image.unsqueeze(0)).squeeze(0)

        mask_binary = torch.zeros(mask.shape).to(device)
        mask_binary[mask > 0.5] = 1

        copyfile(file_name, 'data/test/{:s}.jpg'.format(hash))
        utils.save_image(mask_binary, 'data/test/{:s}_mask.png'.format(hash))
        image *= mask_binary
        utils.save_image(image, 'data/test/{:s}_result.png'.format(hash))