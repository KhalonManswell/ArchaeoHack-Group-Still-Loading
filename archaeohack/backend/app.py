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
import os

from PIL import ImageDraw

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
            
        
        hieroglyph_database = [h for h in data if h.get('description') and h['description'] != ""]
        
        # Create mapping dictionaries
        for idx, h in enumerate(hieroglyph_database):
            gardiner_to_index[h['gardiner_num']] = idx
            index_to_gardiner[idx] = h['gardiner_num']
            
        print(f"Loaded {len(hieroglyph_database)} hieroglyphs (all categories)")
        return True
        
    except Exception as e:
        print(f"Error loading hieroglyph database: {e}")
        return False

def load_model():
    """Load the trained YOLO model"""
    global model, class_names
    
    try:
        # Load YOLO model - try different possible paths
        model_paths = [
            '../model/best.pt',
            '../model/runs/classify/1759/weights/best.pt',
            '../model/1759/weights/best.pt',
            'best.pt'
        ]
        
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

def save_debug_image(image, filename):
    """Save image for debugging purposes"""
    debug_dir = "debug_images"
    os.makedirs(debug_dir, exist_ok=True)
    image_path = os.path.join(debug_dir, filename)
    
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    image.save(image_path)
    print(f"✓ Debug image saved: {image_path}")
    return image_path

def preprocess_image(image_base64):
    """
    Convert base64 image from canvas to format expected by YOLO
    With improved binary conversion to avoid blank images
    """
    try:
        # Remove data URL prefix if present
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        # Decode base64 to image
        image_bytes = base64.b64decode(image_base64)
        original_image = Image.open(io.BytesIO(image_bytes))
        
        print(f"✓ Original image: {original_image.size}, mode: {original_image.mode}")
        
        # Convert to numpy array
        img_array = np.array(original_image)
        print(f"✓ Image array shape: {img_array.shape}, dtype: {img_array.dtype}")
        
        # Convert RGBA to RGB if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            print("🔄 Converting RGBA to RGB with white background...")
            # Create white background
            rgb_img = np.ones((img_array.shape[0], img_array.shape[1], 3), dtype=np.uint8) * 255
            # Use alpha channel to blend
            alpha = img_array[:, :, 3] / 255.0
            for c in range(3):
                rgb_img[:, :, c] = (1 - alpha) * 255 + alpha * img_array[:, :, c]
            img_array = rgb_img.astype(np.uint8)
            print(f"✓ After RGBA conversion: {img_array.shape}")
        
        # Convert to grayscale (matching your training)
        if len(img_array.shape) == 3:
            print("🔄 Converting to grayscale...")
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        print(f"✓ Grayscale image: {gray.shape}, range: [{gray.min()}-{gray.max()}]")
        
        # ===========================
        # IMPROVED BINARY CONVERSION - FIX FOR BLANK IMAGES
        # ===========================
        print("🔄 Applying adaptive binary threshold...")
        
        # Method 1: Check if image is already high contrast
        unique_vals = np.unique(gray)
        print(f"✓ Unique values in grayscale: {unique_vals}")
        
        # If image has few unique values, it might already be binary
        if len(unique_vals) <= 3:
            print("⚠ Image appears to be already binary, using as-is")
            binary = gray
        else:
            # Method 2: Use adaptive threshold instead of fixed threshold
            # This handles different lighting conditions better
            binary_adaptive = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Method 3: Try OTSU automatic threshold
            _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Method 4: Try different fixed thresholds
            _, binary_low = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
            _, binary_medium = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            _, binary_high = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            
            # Check which threshold result has content (not all white or all black)
            binaries = {
                'adaptive': binary_adaptive,
                'otsu': binary_otsu,
                'low_100': binary_low,
                'medium_150': binary_medium,
                'high_200': binary_high
            }
            
            # Select the best binary result (neither all white nor all black)
            best_binary = None
            best_score = -1
            
            for name, bin_img in binaries.items():
                white_pixels = np.sum(bin_img == 255)
                black_pixels = np.sum(bin_img == 0)
                total_pixels = bin_img.size
                
                # Calculate score for non-extreme images (neither all white nor all black)
                if black_pixels > 0.01 * total_pixels and white_pixels > 0.01 * total_pixels:
                    score = min(black_pixels, white_pixels)  # Balance black and white pixels
                    if score > best_score:
                        best_score = score
                        best_binary = bin_img
                        print(f"  - {name}: {black_pixels} black, {white_pixels} white (score: {score})")
            
            if best_binary is not None:
                binary = best_binary
                print(f"✓ Selected best binary: {best_score} score")
            else:
                print("⚠ No suitable binary found, using OTSU")
                binary = binary_otsu
        
        print(f"✓ Binary image: {binary.shape}, range: [{binary.min()}-{binary.max()}]")
        
        # Invert if needed (make drawing black on white background)
        # Check if there's more black than white
        if np.sum(binary == 0) > binary.size * 0.5:
            print("🔄 Inverting binary image (too much black)...")
            binary = cv2.bitwise_not(binary)
        
        # Convert back to PIL Image for YOLO
        binary_img = Image.fromarray(binary)
        
        save_debug_image(binary_img, "preprocessed_binary.png")
        
        return binary_img
        
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None

def test_preprocessing():
    """
    Test endpoint that creates a sample image and processes it
    """
    try:
        # Create a simple test image (a circle)
        test_image = Image.new('RGBA', (400, 400), (255, 255, 255, 0))
        draw = ImageDraw.Draw(test_image)
        draw.ellipse([100, 100, 300, 300], fill=(0, 0, 0, 255))
        
        # Convert to base64
        buffered = io.BytesIO()
        test_image.save(buffered, format="PNG")
        test_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Process it
        processed = preprocess_image(test_base64)
        
        if processed:
            # Convert processed image to base64
            buffered_final = io.BytesIO()
            processed.save(buffered_final, format="PNG")
            processed_base64 = base64.b64encode(buffered_final.getvalue()).decode()
            
            return jsonify({
                'success': True,
                'message': 'Test preprocessing completed',
                'test_image': f"data:image/png;base64,{test_base64}",
                'processed_image': f"data:image/png;base64,{processed_base64}",
                'check_debug_folder': 'Look in debug_images/ folder for step-by-step images'
            })
        else:
            return jsonify({'error': 'Preprocessing failed'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
    
    FIXED VERSION: Properly maps JSON fields to response format
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
        
        print(f"\n🔍 DEBUG: Predicted Gardiner: {predicted_gardiner}")
        print(f"🔍 DEBUG: Confidence: {confidence}%")
        
        # Get hieroglyph info from database
        hieroglyph_info = next((h for h in hieroglyph_database if h['gardiner_num'] == predicted_gardiner), None)
        
        if not hieroglyph_info:
            # Try to find partial match
            hieroglyph_info = next((h for h in hieroglyph_database if predicted_gardiner in h['gardiner_num']), None)
        
        # DEBUG: Log what we found
        if hieroglyph_info:
            print(f"✓ Found in database: {hieroglyph_info['gardiner_num']}")
        else:
            print(f"✗ NOT found in database for: {predicted_gardiner}")
            print(f"  Available priority hieroglyphs: {[h['gardiner_num'] for h in hieroglyph_database[:5]]}...")
        
        if not hieroglyph_info:
            # Fallback: create basic info with actual data
            hieroglyph_info = {
                'gardiner_num': predicted_gardiner,
                'description': f'Hieroglyph {predicted_gardiner}',
                'hieroglyph': '?',
                'unicode_hex': '0000',
                'details': 'Not found in database'
            }
        
        # FIXED: Properly map JSON fields to response format
        response = {
            'sign': hieroglyph_info['gardiner_num'],  # Just the Gardiner number
            'gardiner_num': hieroglyph_info['gardiner_num'],
            'phonetic': hieroglyph_info.get('details', 'No phonetic value available'),
            'meaning': hieroglyph_info.get('description', 'No description available'),
            'unicode': f"U+{hieroglyph_info.get('unicode_hex', '0000').upper()}",
            'glyph': hieroglyph_info.get('hieroglyph', '?'),
            'confidence': round(confidence, 1),
            'additionalInfo': hieroglyph_info.get('details', ''),
            
            # Include top 3 predictions for debugging
            'top_predictions': []
        }
        
        print(f"\n📦 RESPONSE BEING SENT:")
        print(f"  - Sign: {response['sign']}")
        print(f"  - Glyph: {response['glyph']}")
        print(f"  - Unicode: {response['unicode']}")
        print(f"  - Meaning: {response['meaning']}")
        print(f"  - Phonetic: {response['phonetic']}\n")
        
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
        'sample_classes': list(class_names.values())[:10] if isinstance(class_names, dict) else class_names[:10] if class_names else [],
        'sample_hieroglyphs': [
            {
                'gardiner_num': h['gardiner_num'],
                'description': h['description'],
                'unicode_hex': h.get('unicode_hex', 'XXXX'),
                'hieroglyph': h['hieroglyph']
            }
            for h in hieroglyph_database[:3]
        ]
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
