# Sample Backend Integration Example

## Flask API Example (Python)

This is a **complete working example** of how to create a backend that connects to your frontend.

### File: `backend/app.py`

```python
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend to connect

# Load your ML model here
# model = load_your_model('path/to/model')

@app.route('/classify', methods=['POST'])
def classify():
    """
    Receives base64 encoded image from frontend canvas
    Returns hieroglyph classification results
    """
    try:
        # Get the image data from request
        data = request.json
        image_data = data['image']
        
        # Remove the data URL prefix (data:image/png;base64,)
        image_data = image_data.split(',')[1]
        
        # Decode base64 to image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to format your model expects
        # Example: Convert to numpy array
        image_array = np.array(image)
        
        # TODO: Replace this with your actual model prediction
        # prediction = model.predict(preprocess(image_array))
        
        # Mock response - replace with your model's output
        result = {
            'sign': 'water',
            'phonetic': 'n',
            'meaning': 'water, in, of',
            'unicode': 'U+13216',
            'glyph': '𓈖',
            'confidence': 87.5,
            'additionalInfo': 'Common phonetic sign used in many words'
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

---

## Installation

```bash
pip install flask flask-cors pillow numpy
```

---

## Running the Backend

```bash
cd backend
python app.py
```

Server will start at: `http://localhost:5000`

---

## Testing the API

### Using curl:
```bash
curl -X POST http://localhost:5000/classify \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/png;base64,iVBORw0KG..."}'
```

### Using Python:
```python
import requests

response = requests.post('http://localhost:5000/classify', 
    json={'image': 'data:image/png;base64,...'})
print(response.json())
```

---

## With Your Actual ML Model

### Example with TensorFlow/Keras:

```python
import tensorflow as tf
from tensorflow import keras

# Load your trained model
model = keras.models.load_model('path/to/your/model.h5')

# Class labels (hieroglyph names)
CLASSES = ['water', 'mouth', 'reed_leaf', 'vulture', 'house', ...]

def preprocess_image(image):
    """Preprocess image for your model"""
    # Resize to model's expected input size
    image = image.resize((64, 64))  # Adjust to your model
    
    # Convert to grayscale if needed
    image = image.convert('L')
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Normalize pixel values
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        
        # Get the top prediction
        top_index = np.argmax(predictions[0])
        confidence = float(predictions[0][top_index] * 100)
        sign_name = CLASSES[top_index]
        
        # Get hieroglyph data from your dataset
        hieroglyph_data = get_hieroglyph_data(sign_name)
        
        result = {
            'sign': hieroglyph_data['name'],
            'phonetic': hieroglyph_data['phonetic'],
            'meaning': hieroglyph_data['meaning'],
            'unicode': hieroglyph_data['unicode'],
            'glyph': hieroglyph_data['glyph'],
            'confidence': round(confidence, 2),
            'additionalInfo': hieroglyph_data.get('info', '')
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_hieroglyph_data(sign_name):
    """
    Retrieve hieroglyph data from your dataset
    This should query your hieroglyph database/dictionary
    """
    # Example - replace with your actual data source
    hieroglyph_db = {
        'water': {
            'name': 'water',
            'phonetic': 'n',
            'meaning': 'water, in, of',
            'unicode': 'U+13216',
            'glyph': '𓈖',
            'info': 'Common phonetic sign'
        },
        # Add more hieroglyphs...
    }
    
    return hieroglyph_db.get(sign_name, {})
```

---

## Example with PyTorch:

```python
import torch
from torchvision import transforms

# Load model
model = torch.load('path/to/model.pth')
model.eval()

# Define image transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

@app.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Transform image
        image_tensor = transform(image).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        sign_name = CLASSES[predicted.item()]
        confidence_score = confidence.item() * 100
        
        # Get hieroglyph data
        hieroglyph_data = get_hieroglyph_data(sign_name)
        
        result = {
            'sign': hieroglyph_data['name'],
            'phonetic': hieroglyph_data['phonetic'],
            'meaning': hieroglyph_data['meaning'],
            'unicode': hieroglyph_data['unicode'],
            'glyph': hieroglyph_data['glyph'],
            'confidence': round(confidence_score, 2),
            'additionalInfo': hieroglyph_data.get('info', '')
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

---

## Project Structure

```
your-project/
├── frontend/
│   ├── index.html
│   ├── css/
│   └── js/
│       └── app.js         ← Update line ~160 with API URL
├── backend/
│   ├── app.py             ← This file
│   ├── model/
│   │   └── trained_model.h5
│   └── requirements.txt
└── data/
    └── hieroglyphs.json
```

---

## Testing Everything Together

1. **Start backend**:
```bash
cd backend
python app.py
# Server running on http://localhost:5000
```

2. **Start frontend**:
```bash
cd frontend
python -m http.server 8000
# Open http://localhost:8000
```

3. **Draw on canvas and submit** - it should now connect to your ML model!

---

## Troubleshooting

### CORS Error?
Make sure `flask-cors` is installed and enabled:
```python
from flask_cors import CORS
CORS(app)
```

### Model Loading Error?
Check model path and dependencies:
```python
import os
print(os.path.exists('path/to/model'))
```

### Image Format Issues?
Check image size and channels:
```python
print(image.size)  # Should match model input
print(image.mode)  # 'L' for grayscale, 'RGB' for color
```

---

## For Production (Judges Will See This)

### Add error handling:
```python
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500
```

### Add logging:
```python
import logging
logging.basicConfig(level=logging.INFO)

@app.route('/classify', methods=['POST'])
def classify():
    app.logger.info('Received classification request')
    # ... rest of code
```

---

## Quick Integration Checklist

- [ ] Install Flask and dependencies
- [ ] Create `backend/app.py` with this code
- [ ] Load your trained model
- [ ] Update `preprocess_image()` for your model
- [ ] Update `CLASSES` list with your hieroglyphs
- [ ] Test API with curl or Postman
- [ ] Update `frontend/js/app.js` line ~160
- [ ] Test end-to-end from UI
- [ ] Add error handling
- [ ] Write documentation for judges

---

**This example gives you everything you need to connect your ML model to the frontend!**
