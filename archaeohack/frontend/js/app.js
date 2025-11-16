let hieroglyphDatabase = [];
let dataLoaded = false;

// Flask backend URL - change this if your server runs on a different port
const API_URL = 'http://localhost:5000';

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

    // Use all hieroglyphs (removed priority filter)
    const availableHieroglyphs = hieroglyphDatabase;

    if (availableHieroglyphs.length === 0) {
        console.error('No hieroglyphs found in database');
        document.getElementById('promptText').textContent = 'Error: No hieroglyphs available';
        return;
    }
}
    // Randomly select a hieroglyph
    const randomIndex = Math.floor(Math.random() * availableHieroglyphs.length);
    currentPrompt = availableHieroglyphs[randomIndex];
    
    
    // Display the prompt using description as the main prompt
    document.getElementById('promptText').textContent = 
        `"${currentPrompt.description}"`;
    
    console.log(`Generated prompt: ${currentPrompt.description} (${currentPrompt.gardiner_num})`);


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
    
    try {
        // Call the Flask backend
        const response = await fetch(`${API_URL}/classify`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        // Display results based on mode
        if (currentMode === 'identify') {
            displayIdentifyResults(result);
        } else {
            // For practice mode, check if the drawn sign matches the prompt
            const isCorrect = result.gardiner_num === currentPrompt.gardiner_num;
            
            const practiceResult = {
                correct: isCorrect,
                drawnSign: result.sign,
                correctAnswer: {
                    sign: currentPrompt.description,
                    glyph: currentPrompt.hieroglyph
                }
            };
            displayPracticeResults(practiceResult);
        }
    } catch (error) {
        console.error('Error classifying drawing:', error);
        
        // Fallback to mock results if backend is not available
        console.log('Falling back to mock results...');
        document.getElementById('resultsContent').innerHTML = 
            '<p style="color: red; text-align: center;">Error: Backend server not available. Make sure Flask server is running on port 5000.<br><br>Run: python backend/app.py</p>';
    }
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

        <div class="unicode-glyph">
           <div class= "result-value"> ${results.glyph}</div>
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
        
        ${results.top_predictions ? `
            <div class="result-item">
                <div class="result-label">Top 3 Predictions</div>
                <div class="result-value">
                    ${results.top_predictions.map((p, i) => 
                        `${i+1}. ${p.gardiner_num} (${p.confidence}%)`
                    ).join('<br>')}
                </div>
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
