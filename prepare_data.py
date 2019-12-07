import cv2
import time
import numpy as np
from itertools import count
from tqdm import tqdm

def process_image(filename):
    image = cv2.imread(filename)

    edges = cv2.Canny(image, 100, 200)
    edges = cv2.dilate(edges, np.ones((20,20),np.uint8))

    number_of_areas, _, stats, centroids = cv2.connectedComponentsWithStats(edges)

    left = stats[:, 0]
    top = stats[:, 1]
    width = stats[:, 2]
    height = stats[:, 3]
    area = width * height
    ratio = width / height

    area = area.astype(float) / (image.shape[1] * image.shape[0])
    indices = np.argsort(area)[-2::-1]

    meta_index = 0
    while ratio[indices[meta_index]] < 0.6 or ratio[indices[meta_index]] > 3 or (left[indices[meta_index]] + width[indices[meta_index]] / 2) / image.shape[1] > 0.5:
        meta_index += 1

    index = indices[meta_index]
    left, top, width, height = left[index], top[index], width[index], height[index]
    center_x, center_y = left + width // 2, top + height // 2

    size = max(width, height) // 2
    margin = min(center_x - size, center_y - size, image.shape[1] - size - center_x, image.shape[0] - size - center_y, int(size * 0.1))
    size += margin

    cropped = image[center_y-size:center_y+size, center_x-size:center_x+size, :]
    return cropped

for id in count():
    filename = 'data/raw/{:d}.jpg'.format(id)
    cropped = process_image(filename)
    cv2.imwrite('data/images/{:d}.jpg'.format(id), cropped)


cv2.destroyAllWindows()