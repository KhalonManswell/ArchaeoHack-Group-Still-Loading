# Egyptian Hieroglyphs Learning App - Frontend

## 📁 Project Structure

```
frontend/
├── index.html          # Main HTML file
├── css/
│   └── styles.css      # All styling
├── js/
│   └── app.js          # Application logic
└── README.md           # This file
```

## 🚀 How to Run

1. **Simple Method**: Just open `index.html` in your web browser
   - Double-click the file, or
   - Right-click → Open with → Your browser

2. **Local Server Method** (recommended for ML integration):
   ```bash
   # Using Python
   python -m http.server 8000
   # Then open http://localhost:8000
   
   # OR using Node.js
   npx http-server
   ```

## 🎯 Features

### Mode 1: Identify Sign
- Draw a hieroglyph freehand
- Get automatic identification with:
  - Phonetic value(s)
  - Meaning(s)
  - Unicode code point
  - Unicode glyph
  - Confidence score
  - Additional information

### Mode 2: Practice Mode
- Receive a random phonetic value/meaning
- Draw the corresponding hieroglyph
- Get instant feedback (correct/incorrect)
- See the correct answer if wrong

## 🔧 How to Integrate ML Model

### Current State
The app currently uses **mock data** for demonstration. You need to replace the mock functions with your actual ML model.

### Integration Points

Open `js/app.js` and find the section:
```javascript
// ============================================
// ML MODEL INTEGRATION SECTION
// ============================================
```

### Option 1: REST API Integration (Flask/FastAPI Backend)

If your ML model is served via a REST API:

1. **Update the `classifyDrawing` function**:
```javascript
async function classifyDrawing(imageData) {
    try {
        const response = await fetch('http://localhost:5000/classify', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
        });
        
        const result = await response.json();
        return result;
    } catch (error) {
        console.error('Error:', error);
        return null;
    }
}
```

2. **Replace the submit button handler** (around line 128 in app.js):
```javascript
document.getElementById('submitBtn').addEventListener('click', async () => {
    const imageData = canvas.toDataURL('image/png');
    showLoading();
    
    const mlResult = await classifyDrawing(imageData);
    
    if (currentMode === 'identify') {
        displayIdentifyResults(mlResult);
    } else {
        const practiceResult = {
            correct: mlResult.sign === currentPrompt.meaning,
            drawnSign: mlResult.sign,
            correctAnswer: {
                sign: currentPrompt.meaning,
                glyph: mlResult.glyph
            }
        };
        displayPracticeResults(practiceResult);
    }
});
```

3. **Expected API Response Format**:
```json
{
    "sign": "water",
    "phonetic": "n",
    "meaning": "water, in, of",
    "unicode": "U+13216",
    "glyph": "𓈖",
    "confidence": 87.5,
    "additionalInfo": "Common phonetic sign..."
}
```

### Option 2: TensorFlow.js (Browser-Based Model)

If your model is converted to TensorFlow.js:

1. **Add TensorFlow.js to index.html**:
```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
```

2. **Load and use the model in app.js**:
```javascript
let model;

async function loadModel() {
    model = await tf.loadLayersModel('path/to/model.json');
}

async function classifyDrawing() {
    const tensor = tf.browser.fromPixels(canvas);
    // Preprocess tensor as needed
    const prediction = model.predict(tensor);
    // Process and return results
}

window.addEventListener('load', loadModel);
```

## 📝 Mock Functions to Replace

In `js/app.js`, replace these functions:

1. **`getMockResults()`** - Returns dummy identification results
2. **`getMockPracticeResults()`** - Returns dummy practice feedback
3. **`generateNewPrompt()`** - Currently uses hardcoded prompts (connect to your dataset)

## 🎨 UI Customization

### Colors
Edit `css/styles.css` to change the color scheme:
- Primary color: `#667eea` (purple)
- Success: `#51cf66` (green)
- Error: `#ff6b6b` (red)

### Canvas Size
Edit in `index.html`:
```html
<canvas id="drawingCanvas" width="400" height="400"></canvas>
```

## 📦 Dependencies

Currently **zero dependencies**! Just vanilla HTML, CSS, and JavaScript.

If you integrate TensorFlow.js or other libraries, list them here.

## 🐛 Troubleshooting

### Canvas not working
- Make sure JavaScript is enabled in your browser
- Check browser console for errors (F12)

### ML Model not connecting
- Check CORS settings if using external API
- Verify the API endpoint URL is correct
- Check network tab in browser DevTools

### Mobile touch not working
- Touch events are implemented, but test on actual mobile device
- Some browsers may have different touch event handling

## 📊 For Judges

This application is fully functional with:
- ✅ Drawing canvas with mouse/touch support
- ✅ Two modes (Identify Sign & Practice)
- ✅ Clean, intuitive UI
- ✅ Responsive design
- ✅ Ready for ML model integration

The mock data demonstrates the complete user flow and can be replaced with the actual ML model.

## 🔗 Integration with Backend

Your project structure should look like:
```
project-root/
├── frontend/           # This folder
│   ├── index.html
│   ├── css/
│   └── js/
├── backend/            # Your ML model
│   ├── model.py
│   ├── app.py          # Flask/FastAPI server
│   └── trained_model/
└── README.md           # Main project README
```

## 📚 Additional Resources

- [Canvas API Documentation](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API)
- [Fetch API Guide](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)
- [TensorFlow.js Guide](https://www.tensorflow.org/js)

## ✨ Future Enhancements

Potential features to add:
- Save/export drawings
- History of past attempts
- Multiple brush colors
- Undo/redo functionality
- Difficulty levels
- Progress tracking
- Offline mode with localStorage

---

**Created for NYU ArchaeoHack 2025**
