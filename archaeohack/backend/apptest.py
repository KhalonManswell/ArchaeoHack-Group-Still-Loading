from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import base64
import io
import json
import numpy as np
import cv2
import torch
import os

from PIL import ImageDraw

app = Flask(__name__)
CORS(app)

# ===========================
# GLOBAL VARIABLES
# ===========================

model = None
hieroglyph_database = []
gardiner_to_index = {}
index_to_gardiner = {}
class_names = []

# ===========================
# INITIALIZATION FUNCTIONS
# ===========================

def load_hieroglyph_database():
    """Load the hieroglyph database from JSON"""
    global hieroglyph_database, gardiner_to_index, index_to_gardiner
    
    try:
        with open('../frontend/gardiner_hieroglyphs_with_unicode_hex.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        hieroglyph_database = [h for h in data if h.get('is_priority', False)]
        
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
                print(f"✅ Model loaded successfully from {path}")
                break
            except:
                continue
        
        if not model_loaded:
            raise Exception("Could not load model from any path")
        
        if hasattr(model, 'names'):
            class_names = model.names
            print(f"✅ Loaded {len(class_names)} classes from model")
        else:
            print("⚠️ Warning: Could not extract class names from model")
            class_names = {i: h['gardiner_num'] for i, h in enumerate(hieroglyph_database)}
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

# ===========================
# KEY FIX: Improved Image Preprocessing
# ===========================

def preprocess_image(image_base64):
    """
    Improved image preprocessing function
    Key changes:
    1. Remove complex binarization logic
    2. Let YOLO handle image processing (it has internal preprocessing)
    3. Only do basic format conversion
    """
    try:
        # Remove data URL prefix if present
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        # Decode base64 to image
        image_bytes = base64.b64decode(image_base64)
        original_image = Image.open(io.BytesIO(image_bytes))
        
        print(f"📥 Original image: {original_image.size}, mode: {original_image.mode}")
        
        # Convert RGBA to RGB with white background
        if original_image.mode == 'RGBA':
            print("🔄 Converting RGBA to RGB...")
            rgb_image = Image.new('RGB', original_image.size, (255, 255, 255))
            rgb_image.paste(original_image, mask=original_image.split()[3])
            original_image = rgb_image
        elif original_image.mode != 'RGB':
            print("🔄 Converting to RGB...")
            original_image = original_image.convert('RGB')
        
        # ===========================
        # KEY CHANGE: Simplified preprocessing
        # ===========================
        # Option 1: Use original image directly (let YOLO handle preprocessing)
        processed_image = original_image
        
        # Option 2: If training used grayscale, keep consistency
        # Comment out Option 1 and uncomment below code:
        """
        # Convert to grayscale (if your training used grayscale)
        img_array = np.array(original_image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply simple Otsu threshold (automatic threshold selection)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Check if we need to invert (white on black vs black on white)
        white_pixels = np.sum(binary == 255)
        black_pixels = np.sum(binary == 0)
        if white_pixels > black_pixels:
            binary = cv2.bitwise_not(binary)
        
        # Convert back to RGB (YOLO expects RGB)
        processed_image = Image.fromarray(binary).convert('RGB')
        """
        
        print(f"✅ Final preprocessed image: {processed_image.size}, mode: {processed_image.mode}")
        
        # Save debug images (optional)
        save_debug_image(original_image, "01_original.png")
        save_debug_image(processed_image, "02_preprocessed.png")
        
        return processed_image
        
    except Exception as e:
        print(f"❌ Error preprocessing image: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_debug_image(image, filename):
    """Save image for debugging purposes"""
    try:
        debug_dir = "debug_images"
        os.makedirs(debug_dir, exist_ok=True)
        image_path = os.path.join(debug_dir, filename)
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        image.save(image_path)
        print(f"💾 Debug image saved: {image_path}")
    except Exception as e:
        print(f"⚠️ Could not save debug image: {e}")

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
    改进的分类端点
    关键改变：
    1. 使用简化的预处理
    2. 添加更好的错误处理
    3. 改进Unicode字符编码
    """
    try:
        # Get image from request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Preprocess image with improved method
        pil_image = preprocess_image(data['image'])
        if pil_image is None:
            return jsonify({'error': 'Failed to process image'}), 400
        
        # ===========================
        # 关键改变：让YOLO使用正确的图像尺寸
        # ===========================
        # 如果你的训练使用了特定的imgsz，在这里指定
        # 例如：results = model(pil_image, imgsz=200, verbose=False)
        results = model(pil_image, verbose=False)
        
        # Extract predictions
        probs = results[0].probs
        
        if probs is None:
            return jsonify({'error': 'No predictions from model'}), 500
        
        # Get top predictions
        top5_indices = probs.top5
        top5_conf = probs.top5conf
        
        if len(top5_indices) == 0:
            return jsonify({'error': 'No predictions returned'}), 500
        
        # Get the best prediction
        best_idx = int(top5_indices[0])
        confidence = float(top5_conf[0] * 100)
        
        # Get class name (Gardiner number)
        if isinstance(class_names, dict):
            predicted_gardiner = class_names.get(best_idx, f"Class_{best_idx}")
        elif isinstance(class_names, list):
            predicted_gardiner = class_names[best_idx] if best_idx < len(class_names) else f"Class_{best_idx}"
        else:
            predicted_gardiner = f"Class_{best_idx}"
        
        # Clean the gardiner number
        predicted_gardiner = predicted_gardiner.split('.')[0].split('_')[0].upper()
        
        print(f"🎯 Predicted: {predicted_gardiner} with {confidence:.1f}% confidence")
        
        # Get hieroglyph info from database
        hieroglyph_info = next(
            (h for h in hierog
