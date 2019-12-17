import os
import json
from flask import Flask, send_file, jsonify

app = Flask(__name__)

TILE_FILE_FORMAT = 'data/tiles/{:d}/{:d}/{:d}.jpg'

@app.route('/')
def index():
    return send_file('server/index_tiles.html')

@app.route('/tile/<depth>/<x>/<y>.png')
def get_tile(depth, x, y):
    try:
        depth, x, y = int(depth), int(x), int(y)
        return send_file(TILE_FILE_FORMAT.format(depth, x, y))
    except FileNotFoundError:
        return "Tile is empty", 404

app.run(host='0.0.0.0')
