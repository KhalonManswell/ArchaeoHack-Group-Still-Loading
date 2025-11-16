# Egyptian Hieroglyphs Learning App
**NYU ArchaeoHack 2025 Submission**

## Startup (For Judges)

### Prerequisites
- Python 3.8 or higher
- 2GB free disk space
- Modern web browser

### One-Line Setup (Linux/Mac)
```bash
cd archaeohack && pip install -r backend/requirements.txt && python backend/app.py
```

### Windows Setup
```cmd
cd archaeohack
pip install -r backend\requirements.txt
python backend\app.py
```

Then open `frontend/index.html` in your browser.

## 📋 Detailed Installation

### Step 1: Install Dependencies
```bash
cd archaeohack/backend
pip install -r requirements.txt
```
*This will install Flask, PyTorch, and other required packages (~500MB)*

### Step 2: Run the Backend
```bash
python app.py
```
You should see:
```
Model loaded successfully
Starting Flask server...
* Running on http://0.0.0.0:5000
```

### Step 3: Open the Frontend
- Open `frontend/index.html` in any modern browser
- Or use a local server:
  ```bash
  cd frontend
  python -m http.server 8000
  # Visit http://localhost:8000
  ```

## 🎮 How to Use

### Mode 1: Identify Hieroglyph
1. Click "Identify hieroglyph" button
2. Draw a hieroglyph on the canvas
3. Click "Submit Drawing"
4. View the classification results

### Mode 2: Quiz Mode
1. Click "Quiz Mode" button
2. Read the description prompt
3. Draw the requested hieroglyph
4. Click "Submit Drawing"
5. See if you got it correct!

## 🏗️ Architecture

- **Frontend**: Pure HTML/CSS/JavaScript (no build required)
- **Backend**: Flask REST API serving PyTorch model
- **Model**: CNN trained on 171 hieroglyphic signs
- **Database**: JSON file with Gardiner sign list

## 📊 Model Performance
- Training Accuracy: [Add your metrics]
- Validation Accuracy: [Add your metrics]
- Dataset: 171 priority hieroglyphs from Gardiner's sign list

## ⚠️ Troubleshooting

### "Module not found" Error
```bash
pip install --upgrade pip
pip install -r backend/requirements.txt
```

### Port 5000 Already in Use
```bash
# Change port in backend/app.py line 330:
app.run(debug=True, host='0.0.0.0', port=5001)
# Also update frontend/js/app.js line 4:
const API_URL = 'http://localhost:5001';
```

### Low Accuracy
- Draw clearly in the center
- Use medium stroke width
- Draw the complete symbol

## 📁 Project Structure
```
archaeohack/
├── frontend/           # Web interface (no installation needed)
├── backend/           # Flask server + ML model
├── model/             # Trained PyTorch models
└── requirements.txt   # All Python dependencies
```

## 👥 Team
[Add your team members here]

## 📄 License
Educational use for NYU ArchaeoHack 2025