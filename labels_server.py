import os
from flask import Flask, send_file, request
import metadata
import random
import csv
from config import *
import numpy as np
import shutil
from image_loader import ImageDataset

items = metadata.load()

latent_codes_embedded = np.load(LATENT_CODES_EMBEDDED_MOVED_FILE_NAME)
latent_codes_embedded[:, 1] *= -1
image_dataset = ImageDataset()

app = Flask(__name__)

IMAGE_FILE_FORMAT = 'data/images_alpha/{:s}.png'

INDEX_HTML = '''<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="utf-8">
        <title>Dataset Labelling</title>
    </head>

    <body>
        <div id="viewer">
            <div id="image" draggable="false" ondragstart="return false;"></div>
            <div class="hline" style="top:100px;"></div>
            <div class="hline" style="top:150px;"></div>
            <div class="hline" style="top:200px;"></div>
            <div class="hline" style="top:250px;"></div>
            <div class="hline" style="top:300px;"></div>
            <div class="hline" style="top:350px;"></div>
            <div class="hline" style="top:400px;"></div>
            <div class="hline" style="top:450px;"></div>
            <div class="hline" style="top:500px;"></div>
            <div class="hline" style="top:550px;"></div>
            <div class="hline" style="top:600px;"></div>
            <div class="vline" style="left:350px;"></div>
        </div>
        <div id="image_id"></div>
        <button id="btn_load">Skip</button>
        <button id="btn_save">Save</button>
        
        <style>
            #viewer {
                background-color: #bdbdbd;
                position: relative;
                width: 700px;
                height: 700px;
            }

            #image {
                width: 500px;
                height: 500px;
                left: 100px;
                top: 100px;
                position: relative;
                pointer-events: none;
                user-select: none;
                -webkit-user-drag: none;
                background-size: 500px;
            }

            .hline {
                width: 100%;
                height: 2px;
                background-color: rgba(0, 0, 0, 0.3);
                position: absolute;
                left: 0px;
                pointer-events: none;
            }

            .vline {
                height: 100%;
                width: 2px;
                background-color: rgba(0, 0, 0, 0.3);
                position: absolute;
                top: 0px;
                pointer-events: none;
            }


            button {
                margin: 10px;
                padding: 20px 80px;
            }
            
        </style>
        <script>
            var viewer = document.getElementById('viewer');
            var image = document.getElementById('image');
            var btnLoad = document.getElementById('btn_load');
            var btnSave = document.getElementById('btn_save');
            var currentIdDiv = document.getElementById('image_id');
            var currentId = null;
            var currentRotation = 0;

            function loadImage(imageId) {
                currentId = imageId;
                image.style.backgroundImage = 'url(image/' + currentId + '.png)';
                image_id.innerHTML = currentId;
                setRotation(60 * Math.random() - 30);
            }

            function requestImage() {
                var request = new XMLHttpRequest();
                request.open('GET', 'getid');
                request.onload = function() {
                    if (request.status === 200) {
                        loadImage(request.responseText);
                    }
                };
                request.send();
            }

            function setRotation(value) {
                image.style.transform = 'rotate(' + value + 'deg)';
                currentRotation = value;
            }

            var urlId = document.location.href.split('?')[1];
            if (urlId !== undefined) {
                loadImage(urlId);
            } else {
                requestImage();
            }
            btnLoad.onclick = requestImage;

            var mousePressed = false;
            var lastMousePosition = 0;

            function getMousePosition(event) {
                return -Math.atan2(event.clientX - 350, event.clientY - 350) / Math.PI * 180;
            }

            viewer.onmousedown = function(event) {
                mousePressed = true;
                lastMousePosition = getMousePosition(event);
            };
            viewer.onmouseup = function(event) {
                mousePressed = false;
            };
            viewer.onmousemove = function(event) {
                if (mousePressed) {
                    movement = getMousePosition(event) - lastMousePosition;
                    lastMousePosition = getMousePosition(event);
                    setRotation(currentRotation + movement);
                }
            };

            btnSave.onclick = function() {
                var request = new XMLHttpRequest();
                request.open('POST', 'save_rotation?id=' + currentId + '&rotation=' + currentRotation);
                request.onload = function() {
                    if (request.status === 200) {
                        requestImage();
                    }
                };
                request.send();
            };

            document.addEventListener("drag", function( event ) {}, false);
            document.addEventListener("dragover", function( event ) {
                event.preventDefault();
            }, false);

            document.addEventListener('drop', function(event) {
                event.preventDefault();
                var fileName = event.dataTransfer.files[0].name;
                var imageId = fileName.substring(0, fileName.length - 4);
                loadImage(imageId);
            }, false);
        </script>
    </body>
</html>'''

if os.path.exists(ROTATION_DATA_FILENAME):
    rotation_file = open(ROTATION_DATA_FILENAME, 'r')
    reader = csv.reader(rotation_file)
    existing_ids_rotation = set(row[0] for row in reader)
    rotation_file.close
else:
    existing_ids_rotaiton = set()

rotation_file = open(ROTATION_DATA_FILENAME, 'a')

if os.path.exists(QUALITY_DATA_FILENAME):
    quality_file = open(QUALITY_DATA_FILENAME, 'r')
    reader = csv.reader(quality_file)
    existing_ids_quality = set(row[0] for row in reader)
    quality_file.close
else:
    existing_ids_quality = set()

quality_file = open(QUALITY_DATA_FILENAME, 'a')

@app.route('/')
def index():
    return INDEX_HTML

@app.route('/image/<id>.png')
def get_image(id):
    try:
        return send_file(IMAGE_FILE_FORMAT.format(id))
    except FileNotFoundError:
        return "File not found", 404

@app.route('/getid')
def get_id():
    while True:
        item = random.choice(items)
        if os.path.exists(IMAGE_FILE_FORMAT.format(item.image_id)) and item.image_id not in existing_ids_rotation:
            return item.image_id


@app.route('/save_rotation', methods=['POST'])
def save_rotation():
    image_id = request.args.get('id')
    if image_id in existing_ids_rotation:
        print("Skipping duplicate id.")
        return 'ok', 200
    existing_ids_rotation.add(image_id)
    rotation = request.args.get('rotation')
    rotation_file.write('{:s},{:s}\n'.format(image_id, rotation))
    rotation_file.flush()
    return 'ok', 200

@app.route('/save_quality', methods=['POST'])
def save_quality():
    image_id = request.args.get('id')
    print(image_id)
    if image_id in existing_ids_quality:
        print("Skipping duplicate id.")
        return 'ok', 200
    existing_ids_quality.add(image_id)
    quality = request.args.get('quality')
    print(image_id, quality)
    quality_file.write('{:s},{:s}\n'.format(image_id, quality))
    quality_file.flush()
    return 'ok', 200

@app.route('/save_quality_batch', methods=['POST'])
def save_quality_batch():
    x1 = float(request.args.get('x1'))
    x2 = float(request.args.get('x2'))
    y1 = float(request.args.get('y1'))
    y2 = float(request.args.get('y2'))
    quality = request.args.get('quality')

    mask = (latent_codes_embedded[:, 0] > x1) & (latent_codes_embedded[:, 0] < x2) & (latent_codes_embedded[:, 1] > y1) & (latent_codes_embedded[:, 1] < y2)
    indices = np.nonzero(mask)[0]
    ids = [image_dataset.hashes[i] for i in indices]

    for image_id in ids:
        if image_id in existing_ids_quality:
            continue
        existing_ids_quality.add(image_id)
        quality_file.write('{:s},{:s}\n'.format(image_id, quality))
    quality_file.flush()
    return 'ok', 200

@app.after_request
def add_header(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

app.run(host='0.0.0.0')
