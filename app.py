from Flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
import re
import io
import base64
from PIL import Image   

app = Flask(__name__)
model = load_model('handwritten_digit.keras')