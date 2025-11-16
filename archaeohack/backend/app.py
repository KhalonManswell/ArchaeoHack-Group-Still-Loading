from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import os
import base64
import io
import json
import numpy as np
import cv2
import torch

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# ===========================
# GLOBAL VARIABLES
# ===========================

model = None
hieroglyph_database = []
gardiner_to_index = {}
index_to_gardiner = {}
class_names = []  # Will store the class names from YOLO model

# ===========================
# INITIALIZATION FUNCTIONS
# ===========================

def load_hieroglyph_database():
    """Load the hieroglyph database from JSON"""
    global hieroglyph_database, gardiner_to_index, index_to_gardiner
    
    try:
        with open('../frontend/gardiner_hieroglyphs_with_unicode_hex.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Filter priority hieroglyphs (matching your training data)
        hieroglyph_database = [h for h in data if h.get('is_priority', False)]
        
        # Create mapping dictionaries
        for idx, h in enumerate(hieroglyph_database):
            gardiner_to_index[h['gardiner_num']] = idx
            index_to_gardiner[idx] = h['gardiner_num']
            
        print(f"Loaded {len(hieroglyph_database)} priority hieroglyphs")
        return True
        
    except Exception as e:
        print(f"Error loading hieroglyph database: {e}")
        return False

def load_model():
    """Load the trained YOLO model"""
    global model, class_names
    
    try:
        # Load YOLO model - try different possible paths
        """""
        model_paths = [
            '../model/best.pt',
            '../model/runs/classify/1759/weights/best.pt',
            '../model/1759/weights/best.pt',
            'best.pt'
        ]
        """

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        ROOT_DIR = os.path.dirname(BASE_DIR)
        MODEL_PATH = os.path.join(ROOT_DIR, "model", "yolo11s-cls.pt")

        print("BASE_DIR:", BASE_DIR)
        print("ROOT_DIR:", ROOT_DIR)
        print("MODEL_PATH:", MODEL_PATH)


        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
        else:
            print(f"[INFO] Loading model from: {MODEL_PATH}")

        try:
            print(f"Trying to load model from: {MODEL_PATH}")
            model = YOLO(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")  
        except Exception as e:
            raise Exception(f"Failed to load model from {MODEL_PATH}: {e}")
        
        """""
        model_loaded = False
        for path in model_paths:
            try:
                print(f"Trying to load model from: {path}")
                model = YOLO(path)
                model_loaded = True
                print(f"Model loaded successfully from {path}")
                break
            except:
                continue
        
        if not model_loaded:
            raise Exception("Could not load model from any path")
        """

        # Get class names from the model
        # YOLO classification models store class names in model.names
        if hasattr(model, 'names'):
            class_names = model.names
            print(f"Loaded {len(class_names)} classes from model")
            print(f"Classes: {class_names}")
        else:
            print("Warning: Could not extract class names from model")
            # Fallback to using gardiner numbers from database
            class_names = {i: h['gardiner_num'] for i, h in enumerate(hieroglyph_database)}
        
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have the YOLO model file (best.pt) in the correct location")
        return False

# ===========================
# IMAGE PROCESSING
# ===========================

def preprocess_image(image_base64):
    """
    Convert base64 image from canvas to format expected by YOLO
    Matches the preprocessing from your training script
    """
    try:
        # Remove data URL prefix if present
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        # Decode base64 to image
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert RGBA to RGB if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            # Create white background
            rgb_img = np.ones((img_array.shape[0], img_array.shape[1], 3), dtype=np.uint8) * 255
            # Use alpha channel to blend
            alpha = img_array[:, :, 3] / 255.0
            for c in range(3):
                rgb_img[:, :, c] = (1 - alpha) * 255 + alpha * img_array[:, :, c]
            img_array = rgb_img.astype(np.uint8)
        
        # Convert to grayscale (matching your training)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply threshold to create binary image (matching your augment_pic function)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # Resize to 200x200 (matching your training)
        resized = cv2.resize(binary, (200, 200))
        
        # Convert back to PIL Image for YOLO
        pil_image = Image.fromarray(resized)
        
        # If grayscale, convert to RGB (YOLO expects RGB)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        return pil_image
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        import traceback
        traceback.print_exc()
        return None

# ===========================
# API ENDPOINTS
# ===========================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'database_loaded': len(hieroglyph_database) > 0,
        'model_type': 'YOLO11s-cls',
        'num_classes': len(class_names) if class_names else 0
    })

@app.route('/classify', methods=['POST'])
def classify():
    """
    Main classification endpoint using YOLO
    Expects JSON with 'image' field containing base64 encoded image
    """
    try:
        # Get image from request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Preprocess image
        pil_image = preprocess_image(data['image'])
        if pil_image is None:
            return jsonify({'error': 'Failed to process image'}), 400
        
        # Run YOLO prediction
        results = model(pil_image, imgsz=200, verbose=False)
        
        # Extract predictions
        # For classification tasks, YOLO returns probabilities for each class
        probs = results[0].probs  # Get probability object
        
        if probs is None:
            return jsonify({'error': 'No predictions from model'}), 500
        
        # Get top 3 predictions
        top5_indices = probs.top5  # Top 5 class indices
        top5_conf = probs.top5conf  # Top 5 confidences
        
        # Get the best prediction
        best_idx = top5_indices[0] if len(top5_indices) > 0 else 0
        confidence = float(top5_conf[0] * 100) if len(top5_conf) > 0 else 0.0
        
        # Get class name (Gardiner number)
        if isinstance(class_names, dict):
            predicted_gardiner = class_names.get(best_idx, f"Class_{best_idx}")
        elif isinstance(class_names, list):
            predicted_gardiner = class_names[best_idx] if best_idx < len(class_names) else f"Class_{best_idx}"
        else:
            predicted_gardiner = f"Class_{best_idx}"
        
        # Clean the gardiner number (remove any file extensions or extra text)
        # YOLO might have learned from folder names like "A1" or filenames
        predicted_gardiner = predicted_gardiner.split('.')[0].split('_')[0].upper()
        
        # Get hieroglyph info from database
        hieroglyph_info = next((h for h in hieroglyph_database if h['gardiner_num'] == predicted_gardiner), None)
        
        if not hieroglyph_info:
            # Try to find partial match
            hieroglyph_info = next((h for h in hieroglyph_database if predicted_gardiner in h['gardiner_num']), None)
        
        if not hieroglyph_info:
            # Fallback: create basic info
            hieroglyph_info = {
                'gardiner_num': predicted_gardiner,
                'description': f'Hieroglyph {predicted_gardiner}',
                'hieroglyph': '?',
                'unicode_hex': '0000',
                'details': 'Not found in database'
            }
        
        # Prepare response
        response = {
            'sign': f"{hieroglyph_info['gardiner_num']} - {hieroglyph_info['description']}",
            'gardiner_num': hieroglyph_info['gardiner_num'],
            'phonetic': hieroglyph_info.get('details', 'No phonetic value available'),
            'meaning': hieroglyph_info['description'],
            'unicode': f"U+{hieroglyph_info['unicode_hex'].upper()}",
            'glyph': hieroglyph_info['hieroglyph'],
            'confidence': round(confidence, 1),
            'additionalInfo': hieroglyph_info.get('details', ''),
            
            # Include top 3 predictions for debugging
            'top_predictions': []
        }
        
        # Add top 3 predictions
        for i in range(min(3, len(top5_indices))):
            idx = top5_indices[i]
            conf = float(top5_conf[i] * 100)
            
            if isinstance(class_names, dict):
                class_name = class_names.get(idx, f"Class_{idx}")
            elif isinstance(class_names, list):
                class_name = class_names[idx] if idx < len(class_names) else f"Class_{idx}"
            else:
                class_name = f"Class_{idx}"
            
            class_name = class_name.split('.')[0].split('_')[0].upper()
            
            response['top_predictions'].append({
                'gardiner_num': class_name,
                'confidence': round(conf, 1)
            })
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in classification: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/hieroglyphs', methods=['GET'])
def get_hieroglyphs():
    """Get list of all available hieroglyphs"""
    return jsonify({
        'hieroglyphs': [
            {
                'gardiner_num': h['gardiner_num'],
                'description': h['description'],
                'glyph': h['hieroglyph']
            }
            for h in hieroglyph_database
        ],
        'model_classes': class_names if class_names else []
    })

@app.route('/debug', methods=['GET'])
def debug_info():
    """Debug endpoint to check model and data status"""
    return jsonify({
        'model_loaded': model is not None,
        'model_type': type(model).__name__ if model else None,
        'database_size': len(hieroglyph_database),
        'class_names_type': type(class_names).__name__,
        'num_classes': len(class_names) if class_names else 0,
        'sample_classes': list(class_names.values())[:10] if isinstance(class_names, dict) else class_names[:10] if class_names else []
    })

# ===========================
# MAIN EXECUTION
# ===========================

if __name__ == '__main__':
    print("="*50)
    print("Egyptian Hieroglyphs YOLO Classifier")
    print("="*50)
    print("Starting Flask server...")
    
    # Load database and model
    if not load_hieroglyph_database():
        print("Warning: Failed to load hieroglyph database")
        # Continue anyway - model might still work
        
    if not load_model():
        print("ERROR: Failed to load YOLO model. Exiting.")
        print("\nMake sure you have:")
        print("1. Trained the model using train.py")
        print("2. The best.pt file in the model/ directory")
        print("3. ultralytics package installed: pip install ultralytics")
        exit(1)
    
    print("\n" + "="*50)
    print("Server ready!")
    print("="*50)
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
