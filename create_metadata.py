from tqdm import tqdm
import csv
from config import *

class DataProperty():
    def __init__(self, header_name, name, type=str):
        self.header_name = header_name
        self.name = name
        self.values = []
        self.type = type

columns = [
    DataProperty('family', 'Family'),
    DataProperty('genus', 'Genus'),
    DataProperty('specificEpithet', 'Species'),
    DataProperty('infraspecificEpithet', 'Subspecies'),
    DataProperty('higherClassification', 'Higher Classification'),
    DataProperty('sex', 'Sex'),
    DataProperty('decimalLatitude', 'Latitude', float),
    DataProperty('decimalLongitude', 'Longitude', float),
    DataProperty('country', 'Country'),
    DataProperty('scientificName', 'Name'),
    DataProperty('scientificNameAuthorship', 'Name Author'),
    DataProperty('day', 'Day'),
    DataProperty('month', 'Month'),
    DataProperty('year', 'Year'),
    DataProperty('_id', 'id', int),
    DataProperty('occurrenceID', 'Occurence ID'),
]

file = open('data/occurrence.csv', 'r')
reader = csv.DictReader(file)
reader_iterator = iter(reader)

row_by_id = {}

row_index = 0
progress = tqdm(total=1039840, desc='Reading occurence.csv')

for row in reader_iterator:
    id = int(row['_id'])
    progress.update()
    if 'lepidoptera' not in row['higherClassification'].lower():
            continue
    for data_property in columns:
        data_property.values.append(row[data_property.header_name].strip())

    row_by_id[id] = row_index
    row_index += 1

strings = []
name_ids = {}

file = open('data/multimedia.csv', 'r')
reader = csv.DictReader(file)
reader_iterator = iter(reader)

progress = tqdm(total=2126980, desc='Reading multimedia.csv')

image_ids = []

for row in reader_iterator:
    progress.update()
    id = int(row['_id'])
    image = row['identifier'].split('/')[-1]
    title = row['title']
    if '_label_' in title:
        continue
    if row['format'] != 'image/jpeg':
        continue
    if id not in row_by_id:
        continue
    image_ids.append((image, id))

with open(METADATA_FILE_NAME, 'w') as file:
    csv_writer = csv.writer(file, delimiter=',')
    csv_writer.writerow([c.name for c in columns] + ['image'])
    for image, id in tqdm(image_ids, desc='Writing metadata.csv'):
        row_index = row_by_id[id]
        csv_writer.writerow([c.values[row_index] for c in columns] + [image])