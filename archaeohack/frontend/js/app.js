let hieroglyphDatabase = [];
let dataLoaded = false;

// Load hieroglyph data from JSON file
async function loadHieroglyphData() {
    try {
        const response = await fetch('gardiner_hieroglyphs_with_unicode_hex.json');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        
        // Filter out entries with no description (incomplete variants)
        hieroglyphDatabase = data.filter(h => h.description !== null && h.description !== "");
        
        dataLoaded = true;
        console.log(`✓ Loaded ${hieroglyphDatabase.length} hieroglyphs successfully`);
        
        // If we're already in practice mode when data loads, generate a prompt
        if (currentMode === 'practice') {
            generateNewPrompt();
        }
    } catch (error) {
        console.error('Error loading hieroglyph data:', error);
        console.error('Make sure data/gardiner_hieroglyphs.json exists in your project');
        
        // Show error to user
        const promptText = document.getElementById('promptText');
        if (promptText) {
            promptText.textContent = 'Error loading hieroglyph database. Check console for details.';
            promptText.style.color = 'red';
        }
    }
}
// Call this when the page loads
window.addEventListener('load', loadHieroglyphData);

// ============================================
// CANVAS SETUP AND DRAWING FUNCTIONALITY
// ============================================

const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;
let strokeWidth = 5;

// Set up canvas for drawing
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

// Drawing event handlers
function startDrawing(e) {
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left || e.touches[0].clientX - rect.left;
    const y = e.clientY - rect.top || e.touches[0].clientY - rect.top;
    ctx.beginPath();
    ctx.moveTo(x, y);
}

function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left || e.touches[0].clientX - rect.left;
    const y = e.clientY - rect.top || e.touches[0].clientY - rect.top;
    
    ctx.strokeStyle = '#000';
    ctx.lineWidth = strokeWidth;
    ctx.lineTo(x, y);
    ctx.stroke();
}

function stopDrawing() {
    isDrawing = false;
    ctx.beginPath();
}

// Mouse events
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

// Touch events for mobile
canvas.addEventListener('touchstart', startDrawing);
canvas.addEventListener('touchmove', draw);
canvas.addEventListener('touchend', stopDrawing);

// ============================================
// UI CONTROLS
// ============================================

// Stroke width control

/*Stroke width control removed as of now
document.getElementById('strokeWidth').addEventListener('input', (e) => {
    strokeWidth = e.target.value;
    document.getElementById('strokeWidthValue').textContent = strokeWidth + 'px';
});*/

// Clear canvas button
document.getElementById('clearBtn').addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
});

// ============================================
// MODE MANAGEMENT
// ============================================

let currentMode = 'identify'; // 'identify' or 'practice'
let currentPrompt = null;

document.getElementById('identifyModeBtn').addEventListener('click', () => {
    switchMode('identify');
});

document.getElementById('practiceModeBtn').addEventListener('click', () => {
    switchMode('practice');
});

function switchMode(mode) {
    currentMode = mode;
    
    // Update button states
    document.getElementById('identifyModeBtn').classList.toggle('active', mode === 'identify');
    document.getElementById('practiceModeBtn').classList.toggle('active', mode === 'practice');
    
    // Show/hide prompt area
    document.getElementById('promptArea').classList.toggle('hidden', mode === 'identify');
    
    // Clear canvas and results
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    document.getElementById('resultsContent').innerHTML = 
        '<p style="text-align: center; color: #999; padding: 40px;">Draw a hieroglyph and click Submit to see results</p>';
    
    // If switching to practice mode, generate a new prompt
    if (mode === 'practice') {
        generateNewPrompt();
    }
}

// ============================================
// PRACTICE MODE - PROMPT GENERATION
// ============================================

function generateNewPrompt() {
    // Check if data is loaded yet
    if (!dataLoaded || hieroglyphDatabase.length === 0) {
        document.getElementById('promptText').textContent = 'Loading hieroglyphs...';
        console.log('Waiting for hieroglyph data to load...');
        
        // Try again in a moment
        setTimeout(generateNewPrompt, 500);
        return;
    }
    
    // Randomly select a hieroglyph from the database
    const randomIndex = Math.floor(Math.random() * hieroglyphDatabase.length);
    currentPrompt = hieroglyphDatabase[randomIndex];
    
    // Display the prompt using description as the main prompt
    document.getElementById('promptText').textContent = 
        `"${currentPrompt.description}"`;
    
    console.log(`Generated prompt: ${currentPrompt.description} (${currentPrompt.gardiner_num})`);
}
// Find a hieroglyph by its Gardiner code (useful when ML model returns "A1", "N35", etc.)
function findHieroglyphByCode(code) {
    return hieroglyphDatabase.find(h => h.gardiner_num === code) || null;
}

// Get the image path for a hieroglyph
function getHieroglyphImagePath(code) {
    return `images/hieroglyphs/${code}.png`;
}

// Format Unicode hex to standard format
function formatUnicodePoint(hex) {
    return `U+${hex.toUpperCase()}`;
}

// ============================================
// SUBMIT DRAWING FOR CLASSIFICATION
// ============================================

document.getElementById('submitBtn').addEventListener('click', async () => {
    // Check if data is loaded
    if (!dataLoaded || hieroglyphDatabase.length === 0) {
        document.getElementById('resultsContent').innerHTML = 
            '<p style="color: red; text-align: center;">Hieroglyph database still loading. Please wait a moment and try again.</p>';
        return;
    }
    
    // Get canvas image data
    const imageData = canvas.toDataURL('image/png');    
    // Show loading state
    showLoading();
    
    // Simulate ML model processing with timeout
    // When you add your real ML model, replace this setTimeout with your actual API call
    setTimeout(() => {
        try {
            if (currentMode === 'identify') {
                displayIdentifyResults(getMockResults());
            } else {
                displayPracticeResults(getMockPracticeResults());
            }
        } catch (error) {
            console.error('Error displaying results:', error);
            document.getElementById('resultsContent').innerHTML = 
                '<p style="color: red; text-align: center;">Error generating results. Check console for details.</p>';
        }
    }, 1000);
});

function showLoading() {
    document.getElementById('resultsContent').innerHTML = `
        <div class="loading">
            <div class="spinner"></div>
            <p>Analyzing your drawing...</p>
        </div>
    `;
}

// ============================================
// DISPLAY RESULTS
// ============================================

// Display results for Identify mode (Function 1)
function displayIdentifyResults(results) {
    const html = `
        <div class="result-item">
            <div class="result-label">Identified Sign</div>
            <div class="result-value">${results.sign}</div>
        </div>
        
        <div class="result-item">
            <div class="result-label">Phonetic Value</div>
            <div class="result-value">${results.phonetic}</div>
        </div>
        
        <div class="result-item">
            <div class="result-label">Meaning</div>
            <div class="result-value">${results.meaning}</div>
        </div>
        
        <div class="result-item">
            <div class="result-label">Unicode Code Point</div>
            <div class="result-value">${results.unicode}</div>
        </div>
        
        <div class="unicode-glyph">
            ${results.glyph}
        </div>
        
        <div class="result-item">
            <div class="result-label">Confidence</div>
            <div class="result-value">${results.confidence}%</div>
        </div>
        
        ${results.additionalInfo ? `
            <div class="result-item">
                <div class="result-label">Additional Information</div>
                <div class="result-value">${results.additionalInfo}</div>
            </div>
        ` : ''}
    `;
    
    document.getElementById('resultsContent').innerHTML = html;
}

// Display results for Practice mode (Function 2)
function displayPracticeResults(results) {
    const feedbackClass = results.correct ? 'correct' : 'incorrect';
    const feedbackText = results.correct 
        ? '✓ Correct! Well done!' 
        : `✗ Not quite. You drew: "${results.drawnSign}"`;
    
    let html = `
        <div class="feedback ${feedbackClass}">
            ${feedbackText}
        </div>
    `;
    
    if (!results.correct) {
        html += `
            <div class="result-item">
                <div class="result-label">Correct Answer</div>
                <div class="result-value">${results.correctAnswer.sign}</div>
            </div>
            
            <div class="unicode-glyph">
                ${results.correctAnswer.glyph}
            </div>
            
            <div class="result-item">
                <div class="result-label">Your Drawing Was Identified As</div>
                <div class="result-value">${results.drawnSign}</div>
            </div>
        `;
    } else {
        html += `
            <div class="unicode-glyph">
                ${results.correctAnswer.glyph}
            </div>
        `;
    }
    
    html += `
        <button class="btn btn-submit" style="width: 100%; margin-top: 20px;" onclick="nextPractice()">
            Next Question →
        </button>
    `;
    
    document.getElementById('resultsContent').innerHTML = html;
}

// Move to next practice question
function nextPractice() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    generateNewPrompt();
    document.getElementById('resultsContent').innerHTML = 
        '<p style="text-align: center; color: #999; padding: 40px;">Draw the hieroglyph and click Submit</p>';
}

// ============================================
// MOCK FUNCTIONS - REPLACE WITH ML MODEL
// ============================================

function getMockResults() {
    // Safety check
    if (!dataLoaded || hieroglyphDatabase.length === 0) {
        throw new Error('Hieroglyph database not loaded');
    }
    
    // Pick a random hieroglyph from database
    const hieroglyph = hieroglyphDatabase[Math.floor(Math.random() * hieroglyphDatabase.length)];
    
    return {
        sign: `${hieroglyph.hieroglyph} (${hieroglyph.description})`,
        phonetic: hieroglyph.details || "See details",
        meaning: hieroglyph.description,
        unicode: formatUnicodePoint(hieroglyph.unicode_hex),
        glyph: hieroglyph.hieroglyph,
        confidence: Math.floor(Math.random() * 20) + 80, // Random 80-100%
        additionalInfo: hieroglyph.details
    };
}

function getMockPracticeResults() {
    // Safety check
    if (!dataLoaded || hieroglyphDatabase.length === 0) {
        throw new Error('Hieroglyph database not loaded');
    }
    
    // For demo: randomly determine if answer is correct
    const isCorrect = Math.random() > 0.5;
    
    let drawnHieroglyph = currentPrompt;
    if (!isCorrect) {
        // Pick a different random hieroglyph
        do {
            drawnHieroglyph = hieroglyphDatabase[Math.floor(Math.random() * hieroglyphDatabase.length)];
        } while (drawnHieroglyph.gardiner_num === currentPrompt.gardiner_num);
    }
    
    return {
        correct: isCorrect,
        drawnSign: drawnHieroglyph.description,
        correctAnswer: {
            sign: currentPrompt.description,
            glyph: currentPrompt.hieroglyph
        }
    };
}

// ============================================
// ML MODEL INTEGRATION SECTION
// ============================================

/**
 * REPLACE THIS SECTION WITH YOUR ACTUAL ML MODEL INTEGRATION
 * 
 * Option 1: REST API Integration (Flask/FastAPI backend)
 * --------------------------------------------------------
 * async function classifyDrawing(imageData) {
 *     try {
 *         const response = await fetch('http://localhost:5000/classify', {
 *             method: 'POST',
 *             headers: {
 *                 'Content-Type': 'application/json',
 *             },
 *             body: JSON.stringify({ image: imageData })
 *         });
 *         
 *         if (!response.ok) {
 *             throw new Error('Classification failed');
 *         }
 *         
 *         const result = await response.json();
 *         return result;
 *     } catch (error) {
 *         console.error('Error classifying drawing:', error);
 *         return null;
 *     }
 * }
 * 
 * Then update the submit button handler:
 * document.getElementById('submitBtn').addEventListener('click', async () => {
 *     const imageData = canvas.toDataURL('image/png');
 *     showLoading();
 *     
 *     const mlResult = await classifyDrawing(imageData);
 *     
 *     if (mlResult) {
 *         if (currentMode === 'identify') {
 *             displayIdentifyResults(mlResult);
 *         } else {
 *             const practiceResult = {
 *                 correct: mlResult.sign === currentPrompt.meaning,
 *                 drawnSign: mlResult.sign,
 *                 correctAnswer: {
 *                     sign: currentPrompt.meaning,
 *                     glyph: mlResult.glyph
 *                 }
 *             };
 *             displayPracticeResults(practiceResult);
 *         }
 *     } else {
 *         document.getElementById('resultsContent').innerHTML = 
 *             '<p style="color: red; text-align: center;">Error: Could not classify drawing</p>';
 *     }
 * });
 * 
 * 
 * Option 2: TensorFlow.js Integration (browser-based model)
 * ----------------------------------------------------------
 * let model;
 * 
 * // Load model on page load
 * async function loadModel() {
 *     model = await tf.loadLayersModel('path/to/model.json');
 *     console.log('Model loaded successfully');
 * }
 * 
 * // Preprocess canvas for model
 * function preprocessCanvas(canvas) {
 *     // Convert canvas to tensor and preprocess
 *     let tensor = tf.browser.fromPixels(canvas);
 *     // Add any necessary preprocessing (resize, normalize, etc.)
 *     return tensor;
 * }
 * 
 * async function classifyDrawing() {
 *     const tensor = preprocessCanvas(canvas);
 *     const prediction = model.predict(tensor);
 *     // Process prediction and return results
 *     return processedResult;
 * }
 * 
 * // Call loadModel when page loads
 * window.addEventListener('load', loadModel);
 */
