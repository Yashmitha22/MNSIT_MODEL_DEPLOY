from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
import re
import io
import base64
from PIL import Image   

app = Flask(__name__)
model = load_model('handwritten_digit.keras')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image data from the request
        data = request.get_json()
        image_data = data['image']
        
        # Remove the data URL prefix (data:image/png;base64,)
        image_data = re.sub('^data:image/.+;base64,', '', image_data)
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale and resize to 28x28 (MNIST format)
        image = image.convert('L')
        image = image.resize((28, 28))
        
        # Convert to numpy array and normalize
        image_array = np.array(image)
        image_array = image_array.astype('float32') / 255.0
        
        # Reshape for model input (1, 28, 28, 1)
        image_array = image_array.reshape(1, 28, 28, 1)
        
        # Make prediction
        predictions = model.predict(image_array)
        predicted_digit = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        return jsonify({
            'success': True,
            'prediction': int(predicted_digit),
            'confidence': confidence
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)