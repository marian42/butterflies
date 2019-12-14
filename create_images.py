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

from classifier import Classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSIFIER_FILENAME = 'trained_models/classifier.to'

classifier = Classifier()
classifier.cuda()
classifier.load_state_dict(torch.load(CLASSIFIER_FILENAME))
classifier.eval()

file_names = glob.glob('data/raw/**.jpg', recursive=True)

OUTPUT_RESOLUTION = None
MARGIN = 0.1

with torch.no_grad():
    for file_name in tqdm(file_names):
        hash = file_name.split('/')[-1][:-4]
        
        result_file_name = 'data/images' \
            + '' if OUTPUT_RESOLUTION is None else '_{:d}'.format(OUTPUT_RESOLUTION) \
            + '/{:s}.jpg'.format(hash)
        
        if os.path.exists(result_file_name):
            continue

        try:
            image = load_image(file_name).to(device)
        except ValueError:
            print("Could not open {:s}.".format(file_name))
            continue
        image = classifier.apply(image, margin=MARGIN)
        
        if image is None:
            print("Found nothing in {:s}.".format(hash))
            continue
        
        if OUTPUT_RESOLUTION is not None:
            image = (F.adaptive_avg_pool2d(image, (OUTPUT_RESOLUTION, OUTPUT_RESOLUTION))).data
            
        utils.save_image(image, result_file_name)