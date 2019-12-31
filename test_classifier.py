from itertools import count

import torch
from torchvision import utils
import random
import glob
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

while True:
    file_name = random.choice(file_names)
    hash = file_name.split('/')[-1][:-4]

    image = load_image(file_name).to(device)
    image = classifier.apply(image)
    
    if image is None:
        continue
    
    copyfile(file_name, 'data/test/{:s}.jpg'.format(hash))
    utils.save_image(image, 'data/test/{:s}_result.png'.format(hash))