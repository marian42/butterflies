
TILE_SIZE = 256
IMAGE_SIZE = 128 # also ICON_WIDTH in index.html

TILE_DEPTH = 9

DEPTH_OFFSET = 8

SHADOW_RADIUS = 12
SHADOW_VALUE = 0.8

ROTATION_NETWORK_RESOLUTION = 64


TILE_FILE_FORMAT = 'server/tile{:s}/{:d}/{:d}/{:d}.jpg'
CREATE_HQ_TILES = True


META_DATA_FORMAT = 'server/meta/{:d}_{:d}_{:d}.json'
DATAQUADS_PER_FILE = 16 # also in index.html

METADATA_FILE_NAME = 'data/metadata.csv'
ROTATION_DATA_FILENAME = 'data/rotations.csv'
ROTATIONS_CALCULATED_FILENAME = 'data/rotations_calculated.csv'

USE_VARIATIONAL_AUTOENCODER = True
AUTOENCODER_FILENAME = 'trained_models/variational_autoencoder.to' if USE_VARIATIONAL_AUTOENCODER else 'trained_models/autoencoder.to'

TILE_IMAGE_QUALITY = 90