from flask import Flask, request, jsonify, render_template
from tensorflow import keras

import numpy as np
import re
import io
import base64
from PIL import Image   

app = Flask(__name__)
model = keras.models.load_model('handwritten_digit.keras')

@app.route('/')
def home():
    return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True)