import os
import json
from flask import Flask, send_file, jsonify, request
import metadata
import random
import csv

items = metadata.load()

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
            <img id="image"></img>
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

            img {
                width: 500px;
                height: 500px;
                left: 100px;
                top: 100px;
                position: relative;
                pointer-events: none;
                user-select: none;
                -webkit-user-drag: none;
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

            function loadImage() {
                var request = new XMLHttpRequest();
                request.open('GET', 'getid');
                request.onload = function() {
                    if (request.status === 200) {
                        currentId = request.responseText;
                        image.src = 'image/' + currentId + '.png';
                        image_id.innerHTML = currentId;
                        setRotation(60 * Math.random() - 30);
                    }
                };
                request.send();
            }

            function setRotation(value) {
                image.style.transform = 'rotate(' + value + 'deg)';
                currentRotation = value;
            }

            loadImage();
            btnLoad.onclick = loadImage;

            var mousePressed = false;
            var lastMousePosition = 0;

            viewer.onmousedown = function(event) {
                mousePressed = true;
                lastMousePosition = event.clientX;
            };
            viewer.onmouseup = function(event) {
                mousePressed = false;
            };
            viewer.onmousemove = function(event) {
                if (mousePressed) {
                    movement = event.clientX - lastMousePosition;
                    lastMousePosition = event.clientX;
                    setRotation(currentRotation + movement / 5);
                }
            };

            btnSave.onclick = function() {
                var request = new XMLHttpRequest();
                request.open('POST', 'save_rotation?id=' + currentId + '&rotation=' + currentRotation);
                request.onload = function() {
                    if (request.status === 200) {
                        loadImage();
                    }
                };
                request.send();
            };
        </script>
    </body>
</html>'''

rotation_file = open('data/rotations.csv', 'r')
reader = csv.reader(rotation_file)
existing_ids = set(row[0] for row in reader)
rotation_file.close

rotation_file = open('data/rotations.csv', 'a')

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
        if os.path.exists(IMAGE_FILE_FORMAT.format(item.image_id)) or item.image_id in existing_ids:
            return item.image_id


@app.route('/save_rotation', methods=['POST'])
def save_rotation():
    image_id = request.args.get('id')
    rotation = request.args.get('rotation')
    rotation_file.write('{:s},{:s}\n'.format(image_id, rotation))
    rotation_file.flush()
    return 'ok', 200

app.run(host='0.0.0.0')
