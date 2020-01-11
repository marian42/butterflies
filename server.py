import os
import json
from flask import Flask, send_file, jsonify
from config import *

app = Flask(__name__)

@app.route('/')
def index():
    return send_file('server/index.html')

@app.route('/favicon.png')
def get_favicon():
    return send_file('server/favicon.png')

@app.route('/tile/<depth>/<x>/<y>.jpg')
def get_tile(depth, x, y):
    try:
        depth, x, y = int(depth), int(x), int(y)
        return send_file(TILE_FILE_FORMAT.format(depth, x, y))
    except FileNotFoundError:
        return "Tile is empty", 404

@app.route('/meta.json')
def get_tsne():
    return send_file('data/meta.json')

@app.route('/meta/<depth>_<x>_<y>.json')
def get_metadata(depth, x, y):
    try:
        depth, x, y = int(depth), int(x), int(y)
        return send_file(META_DATA_FORMAT.format(depth, x, y))
    except FileNotFoundError:
        return "No metadata for this quad.", 404

app.run(host='0.0.0.0')
