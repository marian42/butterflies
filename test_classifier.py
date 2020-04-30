from itertools import count

import torch
from torchvision import utils
import random
import glob
from shutil import copyfile
from create_images import RawImageDataset

from classifier import Classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSIFIER_FILENAME = 'trained_models/classifier.to'

classifier = Classifier()
classifier.cuda()
classifier.load_state_dict(torch.load(CLASSIFIER_FILENAME))
classifier.eval()

dataset = RawImageDataset()
dataset.skip_existing_files = False

while True:
    index = random.randint(0, len(dataset))
    hash = dataset.hashes[index]
    image, _, width, height = dataset[index]
    mask = classifier.apply(image.to(device))
    
    if image is None:
        continue
    
    mask = mask[:height, :width]
    
    copyfile('data/raw/{:s}.jpg'.format(hash), 'data/test/{:s}.jpg'.format(hash))
    utils.save_image(mask, 'data/test/{:s}_result.jpg'.format(hash))