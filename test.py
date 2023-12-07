import cv2
import os
import glob
import numpy as np
import time
import pickle
from Facerec import Facerec
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)
sfr = Facerec()
score = 0
count = 0


@app.route("/api/load", methods=["POST"])
def load_face():
    return sfr.load_encoding_images("/saved")


@app.route("/api/check", methods=["POST"])
def check_face():
    img = cv2.imread("/check_path/image.jpg")
    score += sfr.detect_known_faces(img)
    count += 1
    return True


@app.route("/api/eval", methods=["POST"])
def eval():
    return (score / count) * 100


@app.route("/api/reset", methods=["POST"])
def reset():
    return sfr.clear_encodings()
