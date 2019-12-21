import json
from tqdm import tqdm
from collections import Counter
import csv
import math
import numpy as np

class DataProperty():
    def __init__(self, column, name, type=str):
        self.column = column
        self.name = name
        self.values = []
        self.type = type

columns = [
    DataProperty(0, 'id', int),
    DataProperty(51, 'Occurence ID'),
    DataProperty(10, 'Latitude', float),
    DataProperty(11, 'Longitude', float),
    DataProperty(17, 'Family'),
    DataProperty(20, 'Genus'),
    DataProperty(62, 'Species'),
    DataProperty(31, 'Subspecies'),
    DataProperty(61, 'Sex'),
    DataProperty(8, 'Country'),
]

file = open('data/occurrence.csv', 'r')
reader = csv.reader(file)
reader_iterator = iter(reader)
column_names = next(reader_iterator)

times = []
names = []

MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

row_by_id = {}

row_index = 0
progress = tqdm(total=1039840, desc='Reading occurence.csv')

def try_make_int(string):
    try:
        return int(string)
    except:
        return None

def get_time(row):    
    day = try_make_int(row[9])
    month = try_make_int(row[50])
    year = try_make_int(row[70])

    try:
        if day is not None and month is not None and year is not None:
            return '{:s} {:d}, {:d}'.format(MONTHS[month], day, year)
        elif month is not None and year is not None:
            return '{:s} {:d}'.format(MONTHS[month - 1], year)
        elif year is not None:
            return str(year)
        else:
            return None
    except:
        return None

for row in reader_iterator:
    id = int(row[0])
    higher_classification = row[25]
    progress.update()
    if 'papilionoidea' not in higher_classification.lower():
        continue

    for data_property in columns:
        try:
            data_property.values.append(data_property.type(row[data_property.column].strip()))
        except ValueError:
            data_property.values.append(None)   

    name = row[59]
    name_author = row[60]
    if name.endswith(name_author):
        name = name[:-len(name_author)].strip()
    names.append(name)
    times.append(get_time(row))

    row_by_id[id] = row_index
    row_index += 1

strings = []
name_ids = {}

def get_name_id(value):
    if value not in name_ids:
        name_ids[value] = len(strings)
        strings.append(value)
    return name_ids[value]

# Rights: CC BY 4.0 The Trustees of the Natural History Museum, London

data = json.load(open('data/clusters.json', 'r'))

file = open('data/multimedia.csv', 'r')
reader = csv.reader(file)
reader_iterator = iter(reader)
column_names = next(reader_iterator)

progress = tqdm(total=2126980, desc='Reading multimedia.csv')

row_by_image = dict()

for row in reader_iterator:
    id = int(row[0])
    image = row[5].split('/')[-3]
    if id in row_by_id:
        row_by_image[image] = row_by_id[id]
    progress.update()

def set_item_data(item, row):
    item['id'] = columns[0].values[row]
    item['occId'] = columns[1].values[row]
    if columns[2].values[row] is not None and columns[3].values[row] is not None:
        item['lat'] = columns[2].values[row]
        item['lon'] = columns[3].values[row]
    if times[row] is not None:
        item['time'] = times[row]
    properties = [c.values[row] for c in columns[4:]]
    item['properties'] = [get_name_id(v) for v in properties + [names[row]]]


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

result = {}
for depth_str in data:
    depth = int(depth_str)
    items = data[depth_str]   

    quad_count = 2**(depth - 9)

    quads = {}
    
    for item in items:
        item['y'] *= -1

        row = row_by_image[item['image']]
        
        set_item_data(item, row)

        quad_x = math.floor(item['x'] * quad_count)
        quad_y = math.floor(item['y'] * quad_count)

        if quad_x not in quads:
            quads[quad_x] = {}
        if quad_y not in quads[quad_x]:
            quads[quad_x][quad_y] = []
        quads[quad_x][quad_y].append(item)
    
    if depth < 13:
        result[depth] = quads
    else:
        save_dataquads(quads, depth)                

from image_loader import ImageDataset
dataset = ImageDataset()
codes = np.load('data/latent_codes_embedded_moved.npy')
min_value = np.min(codes, axis=0)
max_value = np.max(codes, axis=0)
codes -= (max_value + min_value) / 2
codes /= np.max(codes, axis=0)

final_depth = max(int(d) for d in data.keys()) + 1
quads = {}
quad_count = 2**(final_depth - 9)

for i in range(codes.shape[0]):
    hash = dataset.hashes[i]
    x = codes[i, 0]
    y = -codes[i, 1]

    item = {'x': x, 'y': y, 'image': hash}
    row = row_by_image[hash]
    set_item_data(item, row)

    quad_x = math.floor(item['x'] * quad_count)
    quad_y = math.floor(item['y'] * quad_count)

    if quad_x not in quads:
        quads[quad_x] = {}
    if quad_y not in quads[quad_x]:
        quads[quad_x][quad_y] = []
    quads[quad_x][quad_y].append(item)

save_dataquads(quads, final_depth)

result['names'] = strings
json_string = json.dumps(result)
with open('data/meta.json', 'w') as file:
    file.write(json_string)
