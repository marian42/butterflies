import json
import math
import numpy as np
import metadata
from config import *
import csv

butterflies_by_image_id = {i.image_id: i for i in metadata.load()}

strings = []
name_ids = {}

rotation_file = open('data/rotations_calculated.csv', 'r')
reader = csv.reader(rotation_file)
rotations = {row[0]: float(row[1]) for row in reader}
rotation_file.close()

def get_name_id(value):
    if value not in name_ids:
        name_ids[value] = len(strings)
        strings.append(value)
    return name_ids[value]

def create_json_dict(item, x, y):
    result = {
        'x': x,
        'y': y,
        'occId': item.occurence_id,
        'image': item.image_id,
        'properties': [get_name_id(p) for p in (item.family, item.genus, item.species, item.subspecies, item.sex, item.country, item.pretty_name)]
    }

    if image_id in rotations:
        rotation = int(-rotations[item.image_id] / 90 + 4.25) % 4
        if rotation != 0:
            result['rot'] = rotation

    if item.latitude != '':
        result['lat'] = float(item.latitude)
    if item.longitude != '':
        result['lon'] = float(item.longitude)
    if item.pretty_time is not None:
        result['time'] = item.pretty_time

    return result


class DataQuads():
    def __init__(self, depth):
        self.depth = depth
        self.quad_count = 2**(depth - 9)
        self.quads = {}

    def insert(self, item):
        x, y = item['x'], item['y']

        quad_x = math.floor(x * self.quad_count)
        quad_y = math.floor(y * self.quad_count)

        if quad_x not in self.quads:
            self.quads[quad_x] = {}
        if quad_y not in self.quads[quad_x]:
            self.quads[quad_x][quad_y] = []
        
        self.quads[quad_x][quad_y].append(item)

    def save(self):
        dataquad_files = {}
        for x in self.quads:
            file_x = x // DATAQUADS_PER_FILE
            for y in self.quads[x]:
                file_y = y // DATAQUADS_PER_FILE
                if (file_x, file_y) not in dataquad_files:
                    dataquad_files[(file_x, file_y)] = {}
                dataquad_file = dataquad_files[(file_x, file_y)]
                if x not in dataquad_file:
                    dataquad_file[x] = {}
                dataquad_file[x][y] = self.quads[x][y]
        for file_x, file_y in dataquad_files:
            json_string = json.dumps(dataquad_files[(file_x, file_y)])
            with open(META_DATA_FORMAT.format(self.depth, file_x, file_y), 'w') as file:
                file.write(json_string)

data = json.load(open('data/clusters.json', 'r'))

result = {}
for depth_str in data:
    items = data[depth_str]
    depth = int(depth_str)
    quads = DataQuads(depth)
    
    for item in items:
        x, y, image_id = item['x'], -item['y'], item['image']

        if image_id not in butterflies_by_image_id:
            continue

        quads.insert(create_json_dict(butterflies_by_image_id[image_id], x, y))
    
    if depth < 13:
        result[depth] = quads.quads
    else:
        quads.save()

from image_loader import ImageDataset
dataset = ImageDataset()
codes = np.load('data/latent_codes_embedded_moved.npy')

final_depth = max(int(d) for d in data.keys()) + 1

final_depth_quads = DataQuads(final_depth)

for i in range(codes.shape[0]):
    x, y, image_id = codes[i, 0], -codes[i, 1], dataset.hashes[i]
        
    if image_id not in butterflies_by_image_id:
        continue

    final_depth_quads.insert(create_json_dict(butterflies_by_image_id[image_id], x, y))

final_depth_quads.save()

result['names'] = strings
json_string = json.dumps(result)
with open('server/meta.json', 'w') as file:
    file.write(json_string)
