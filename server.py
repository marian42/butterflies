import os
import json
from flask import Flask, send_file, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return send_file('server/index.html')

@app.route('/tsne.json')
def get_tsne():
    return send_file('data/tsne.json')

@app.route('/image/<hash>')
def get_image(hash):
    return send_file('data/images_128/{:s}.jpg'.format(hash))

app.run()
