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
    DataProperty(17, 'Family'),
    DataProperty(20, 'Genus'),
    DataProperty(62, 'Species'),
    DataProperty(31, 'Subspecies'),
    DataProperty(61, 'Sex'),
    DataProperty(10, 'Latitude', float),
    DataProperty(11, 'Longitude', float),
    DataProperty(8, 'Country'),
    DataProperty(59, 'Name'),
    DataProperty(60, 'Name Author'),
    DataProperty(9, 'Day'),
    DataProperty(50, 'Month'),
    DataProperty(70, 'Year'),
    DataProperty(0, 'id', int),
    DataProperty(51, 'Occurence ID'),
]

file = open('data/occurrence.csv', 'r')
reader = csv.reader(file)
reader_iterator = iter(reader)
column_names = next(reader_iterator)

row_by_id = {}

row_index = 0
progress = tqdm(total=1039840, desc='Reading occurence.csv')

for row in reader_iterator:
    id = int(row[0])
    progress.update()
    if 'lepidoptera' not in row[25].lower():
            continue
    for data_property in columns:
        data_property.values.append(row[data_property.column].strip())

    row_by_id[id] = row_index
    row_index += 1

strings = []
name_ids = {}

file = open('data/multimedia.csv', 'r')
reader = csv.reader(file)
reader_iterator = iter(reader)
column_names = next(reader_iterator)

progress = tqdm(total=2126980, desc='Reading multimedia.csv')

image_ids = []

for row in reader_iterator:
    progress.update()
    id = int(row[0])
    image = row[5].split('/')[-3]
    title = row[2]
    if '_label_' in title:
        continue
    if row[3] != 'image/jpeg':
        continue
    if id not in row_by_id:
        continue
    image_ids.append((image, id))

with open('data/metadata.csv', 'w') as file:
    csv_writer = csv.writer(file, delimiter=',')
    csv_writer.writerow([c.name for c in columns] + ['image'])
    for image, id in tqdm(image_ids, desc='Writing metadata.csv'):
        row_index = row_by_id[id]
        csv_writer.writerow([c.values[row_index] for c in columns] + [image])