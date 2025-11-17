# Egyptian Hieroglyphs Classifier

An interactive deep learning application for recognizing and learning ancient Egyptian hieroglyphic signs. Users can draw hieroglyphs freehand to receive instant classification with phonetic values, meanings, and Unicode information. The application also includes a practice mode for testing knowledge of hieroglyphic symbols.

## Features

- **Identify Mode**: Draw a hieroglyph and receive instant classification with:
  - Gardiner number and Unicode code point
  - Phonetic value and semantic meaning
  - Confidence score
  - Top 3 predictions for verification

- **Practice Mode**: Test your knowledge by drawing hieroglyphs corresponding to random prompts with automatic grading

- **Cross-platform Support**: Works on desktop browsers with mouse, touch input, and graphic tablets

- **Flexible Architecture**: Designed to be easily retrained for other ancient writing systems (Cuneiform, Maya hieroglyphs, etc.)

## System Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended for YOLO model inference)
- Modern web browser with canvas support
- Internet connection for initial setup only

## Installation

### Step 1: Clone or extract the project

Navigate to your project directory in the terminal.

### Step 2: Install dependencies

Run the following command to automatically install all required packages:

```bash
pip install -e .
```

This reads the `pyproject.toml` file and installs all dependencies. No virtual environment activation is required.

### Note for macOS Users

If you encounter an "externally-managed-environment" error, create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .

#### Alternative: Using requirements.txt

If you prefer the traditional approach:

```bash
pip install -r requirements.txt
```

### Step 3: Verify installation

To verify that all dependencies installed correctly:

```bash
python3 -c "import flask, torch, ultralytics, cv2; print('All dependencies installed successfully')"
```

## Running the Application

The application consists of two components that must run in separate terminal windows.

### Terminal 1: Start the Flask Backend API Server

```bash
python3 backend/app.py
```

You should see output indicating:
- Model loaded successfully
- Database loaded with X hieroglyphs
- Server running on http://0.0.0.0:5000

The backend will remain running. Do not close this terminal.

### Terminal 2: Start the Frontend Web Server

```bash
cd frontend
python3 -m http.server 8000
```

You should see:
- Serving HTTP on :: port 8000

### Step 3: Access the Application

Open your web browser and navigate to:

```
http://localhost:8000
```

The application interface should load, displaying the drawing canvas and mode selector buttons.

## Usage

### Identify Mode (Default)

1. Select "Identify Sign Mode" (should be active by default)
2. Draw a hieroglyph on the canvas using your mouse, touchscreen, or stylus
3. Click "Submit Drawing" when finished
4. Results will display:
   - Identified sign (Gardiner number)
   - Unicode glyph rendering
   - Phonetic value(s)
   - Semantic meaning
   - Confidence percentage
   - Top 3 alternative predictions

### Practice Mode

1. Click "Practice Mode" button
2. A random hieroglyph prompt will appear (e.g., "Seated man")
3. Draw the hieroglyph you believe corresponds to that prompt
4. Click "Submit Drawing"
5. The application will provide feedback:
   - "Correct!" if your drawing matches the expected sign
   - "Not quite" with the correct answer if incorrect
6. Click "Next Question" to continue practicing

### Controls

- **Clear Canvas**: Removes your drawing without submitting
- **Submit Drawing**: Sends drawing to backend for classification
- **Mode Buttons**: Switch between Identify and Practice modes

## Project Structure

```
project-root/
├── backend/
│   └── app.py                                    # Flask API server
├── frontend/
│   ├── index.html                               # Main UI
│   ├── app.js                                   # Client-side logic
│   ├── css/
│   │   └── styles.css                          # Styling
│   ├── gardiner_hieroglyphs_with_unicode_hex.json  # Hieroglyph database
│   └── best.pt                                  # Trained YOLO model
├── pyproject.toml                              # Project configuration
├── requirements.txt                            # Dependencies (alternative)
└── README.md                                   # This file
```

## Troubleshooting

### Issue: "Backend server not available" error in browser

**Cause**: Flask backend is not running on port 5000.

**Solution**:
1. Check that Terminal 1 is running `python3 backend/app.py`
2. Verify output shows "Server ready!"
3. Ensure no other application is using port 5000

To check if port 5000 is in use:

**On Windows:**
```bash
netstat -ano | findstr :5000
```

**On Mac/Linux:**
```bash
lsof -i :5000
```

If a process is using port 5000, either close that application or modify `app.py` to use a different port (e.g., 5001) and update `frontend/app.js` line 5:
```javascript
const API_URL = 'http://localhost:5001';
```

### Issue: "No image provided" or HTTP 400 error

**Cause**: The drawing canvas is empty or the submission failed.

**Solution**:
1. Ensure you have drawn on the canvas (visible black lines)
2. Click "Clear Canvas" and try again
3. Check browser console for JavaScript errors (F12 > Console tab)
4. Verify the canvas size is at least 50x50 pixels

### Issue: Question mark (?) displayed instead of hieroglyph glyph

**Cause**: The predicted Gardiner number is not found in the database.

**Solution**:
1. Check backend console output for the predicted Gardiner number
2. Verify `gardiner_hieroglyphs_with_unicode_hex.json` exists in the frontend folder
3. Ensure the JSON file contains entries with descriptions (incomplete variants are filtered out)
4. The YOLO model may need retraining if predictions are consistently incorrect

### Issue: U+0000 displayed instead of Unicode code point

**Cause**: Same as above — hieroglyph not found in database, returning default fallback values.

**Solution**: Follow the steps above for the question mark issue.

### Issue: Very low confidence scores (< 10%)

**Cause**: The YOLO model may be undertrained or the drawing style differs significantly from training data.

**Solution**:
1. Try drawing more clearly and in the center of the canvas
2. Use consistent stroke thickness
3. Ensure drawing matches typical hieroglyph proportions
4. Consider retraining the YOLO model with more diverse training data

### Issue: "ModuleNotFoundError" for specific packages

**Cause**: Dependencies did not install correctly.

**Solution**:
1. Verify you ran `pip install -e .` in the project root
2. Check Python version is 3.8 or higher: `python --version`
3. Try reinstalling with verbose output: `pip install -e . -v`
4. If specific package fails, install manually:
   ```bash
   pip install flask flask-cors torch ultralytics opencv-python
   ```

### Issue: YOLO model file "best.pt" not found

**Cause**: The trained model file is missing from the project.

**Solution**:
1. Verify `best.pt` exists in `frontend/` directory
2. Check backend console for the exact path it's searching
3. Ensure model file is not corrupted (should be > 50MB)
4. If missing, the model must be retrained using your training pipeline

### Issue: "Address already in use" for port 8000

**Cause**: Another process is using port 8000.

**Solution**:
1. Stop the other application
2. Or use an alternative port:
   ```bash
   python3 -m http.server 9000
   ```
   Then access at `http://localhost:9000`

### Issue: Drawing appears but no response after clicking Submit

**Cause**: Backend is processing but taking longer than expected, or there is a communication error.

**Solution**:
1. Check the backend terminal for error messages
2. Verify network connectivity between frontend and backend
3. Try a simpler drawing with fewer strokes
4. Restart both the backend and frontend servers

### Issue: Browser console shows CORS errors

**Cause**: Cross-Origin Resource Sharing is not properly configured between frontend and backend.

**Solution**:
1. Verify Flask backend is running on port 5000
2. Ensure `flask-cors` is installed: `pip install flask-cors`
3. Check that `CORS(app)` is enabled in the backend app.py file
4. Restart the backend server

## Technical Details

### Model Architecture

The application uses YOLOv8 (You Only Look Once v8) for image classification. The model is trained to recognize 769 different Egyptian hieroglyphic signs from the Gardiner classification system.

### Database

Hieroglyphs are stored in `gardiner_hieroglyphs_with_unicode_hex.json` with the following information for each sign:
- Gardiner number (e.g., A1, V38)
- Unicode hex value
- Unicode glyph character
- Description/meaning
- Phonetic value or linguistic details

### API Endpoints

**POST /classify**
- Accepts base64 encoded canvas image
- Returns: Gardiner number, glyph, Unicode value, phonetic value, meaning, confidence score, and top 3 predictions

**GET /health**
- Returns server status, model load status, database size

**GET /debug**
- Returns diagnostic information for troubleshooting

## Performance Considerations

- First prediction may take 3-5 seconds as the model loads into memory
- Subsequent predictions typically respond in under 1 second
- Drawing larger and bolder hieroglyphs improves recognition accuracy
- The application is optimized for drawings on the canvas (400x400 pixels)

## Future Enhancements

Potential improvements for future versions:
- Real-time prediction as user draws
- Undo/redo functionality
- Drawing history tracking
- Support for multiple handwriting styles through transfer learning
- Export results to PDF or image format
- Mobile application version
- Integration with academic databases for additional hieroglyph context

## Support

For issues or questions regarding the application, check the Troubleshooting section above. For technical questions about YOLO model training or accuracy improvements, refer to the official Ultralytics documentation at https://docs.ultralytics.com/
