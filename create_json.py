import json
import math
import numpy as np
import metadata

butterflies_by_image_id = {i.image_id: i for i in metadata.load()}

strings = []
name_ids = {}

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
        'time': item.pretty_time,
        'properties': [get_name_id(p) for p in (item.family, item.genus, item.species, item.subspecies, item.sex, item.country, item.pretty_name)]
    }

    if item.latitude != '':
        result['lat'] = float(item.latitude)
    if item.longitude != '':
        result['lon'] = float(item.longitude)

    return result

DATAQUADS_PER_QUAD = 16

def save_dataquads(quads, depth):
    dataquads = {}
    for x in quads:
        data_x = x // DATAQUADS_PER_QUAD
        for y in quads[x]:
            data_y = y // DATAQUADS_PER_QUAD
            if (data_x, data_y) not in dataquads:
                dataquads[(data_x, data_y)] = {}
            dataquad = dataquads[(data_x, data_y)]
            if x not in dataquad:
                dataquad[x] = {}
            dataquad[x][y] = quads[x][y]
    for data_x, data_y in dataquads:
        json_string = json.dumps(dataquads[(data_x, data_y)])
        with open('data/meta/{:d}_{:d}_{:d}.json'.format(depth, data_x, data_y), 'w') as file:
            file.write(json_string)

data = json.load(open('data/clusters.json', 'r'))

result = {}
for depth_str in data:
    depth = int(depth_str)
    items = data[depth_str]

    quad_count = 2**(depth - 9)

    quads = {}
    
    for item in items:
        x, y, image_id = item['x'], -item['y'], item['image']

        if image_id not in butterflies_by_image_id:
            continue

        json_dict = create_json_dict(butterflies_by_image_id[image_id], x, y)

        quad_x = math.floor(x * quad_count)
        quad_y = math.floor(y * quad_count)

        if quad_x not in quads:
            quads[quad_x] = {}
        if quad_y not in quads[quad_x]:
            quads[quad_x][quad_y] = []
        
        quads[quad_x][quad_y].append(json_dict)
    
    if depth < 13:
        result[depth] = quads
    else:
        save_dataquads(quads, depth)

from image_loader import ImageDataset
dataset = ImageDataset()
codes = np.load('data/latent_codes_embedded_moved.npy')

final_depth = max(int(d) for d in data.keys()) + 1
quads = {}
quad_count = 2**(final_depth - 9)

for i in range(codes.shape[0]):
    x, y, image_id = codes[i, 0], -codes[i, 1], dataset.hashes[i]
        
    if image_id not in butterflies_by_image_id:
        continue

    json_dict = create_json_dict(butterflies_by_image_id[image_id], x, y)

    quad_x = math.floor(x * quad_count)
    quad_y = math.floor(y * quad_count)

    if quad_x not in quads:
        quads[quad_x] = {}
    if quad_y not in quads[quad_x]:
        quads[quad_x][quad_y] = []
    quads[quad_x][quad_y].append(json_dict)

save_dataquads(quads, final_depth)

result['names'] = strings
json_string = json.dumps(result)
with open('data/meta.json', 'w') as file:
    file.write(json_string)
