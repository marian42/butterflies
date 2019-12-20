import json
from tqdm import tqdm
from collections import Counter
import csv

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

# Name:
# Class (Insecta) > Order (Lepidoptera) > Family (Papilionidae) > Genus > Species / Specific Epithet > Subspecies / Infraspecific Epithet

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

for depth in data:
    items = data[depth]
    
    for item in tqdm(items, desc="Depth {:s}".format(depth)):
        item['y'] *= -1
        row = row_by_image[item['image']]
        item['id'] = columns[0].values[row]
        item['occId'] = columns[1].values[row]
        if columns[2].values[row] is not None and columns[3].values[row] is not None:
            item['lat'] = columns[2].values[row]
            item['lon'] = columns[3].values[row]
        if times[row] is not None:
            item['time'] = times[row]
        properties = [c.values[row] for c in columns[4:]]
        item['properties'] = [get_name_id(v) for v in properties + [names[row]]]
        
data['names'] = strings


json_string = json.dumps(data)
with open('data/tsne.json', 'w') as file:
    file.write(json_string)
