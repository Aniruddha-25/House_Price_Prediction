/**
 * House Price Prediction - Frontend JavaScript
 * Handles all client-side interactions and API calls
 */

// Configuration
const API_BASE = '/api';
const USD_TO_INR = 83.5; // Exchange rate
const CONFIG = {
    trainingProgressInterval: 100,
    notificationDuration: 5000,
};

// Chart instances removed - no charts needed

// State management
const state = {
    isTraining: false,
    modelTrained: false,
    trainingProgress: 0,
    analyticsData: null,
};

/**
 * Format number as INR currency
 */
function formatINR(value) {
    return new Intl.NumberFormat('en-IN', {
        style: 'currency',
        currency: 'INR',
        minimumFractionDigits: 0
    }).format(value * USD_TO_INR);
}

/**
 * Show notification message
 */
function showNotification(message, type = 'info') {
    const notification = document.getElementById('notification');
    notification.textContent = message;
    notification.className = `notification ${type}`;
    
    setTimeout(() => {
        notification.classList.add('hidden');
    }, CONFIG.notificationDuration);
}

/**
 * Update model status display
 */
function updateModelStatus(trained = false) {
    const statusEl = document.getElementById('modelStatus');
    
    if (trained) {
        statusEl.className = 'model-status status-trained';
        statusEl.innerHTML = '<span class="status-indicator"></span><span class="status-text">Model Status: Trained ✓</span>';
        state.modelTrained = true;
    } else {
        statusEl.className = 'model-status status-untrained';
        statusEl.innerHTML = '<span class="status-indicator"></span><span class="status-text">Model Status: Not Trained</span>';
        state.modelTrained = false;
    }
}

/**
 * Train the machine learning model
 */
async function trainModel() {
    if (state.isTraining) {
        showNotification('Training already in progress...', 'warning');
        return;
    }
    
    const trainBtn = document.getElementById('trainBtn');
    const trainingProgress = document.getElementById('trainingProgress');
    const trainingResult = document.getElementById('trainingResult');
    const progressText = document.getElementById('progressText');
    
    // Show progress UI
    trainBtn.disabled = true;
    trainingProgress.classList.remove('hidden');
    trainingResult.classList.add('hidden');
    
    state.isTraining = true;
    state.trainingProgress = 0;
    
    // Simulate progress
    const progressInterval = setInterval(() => {
        if (state.trainingProgress < 85) {
            state.trainingProgress += Math.random() * 20;
            if (state.trainingProgress > 85) state.trainingProgress = 85;
        }
    }, 500);
    
    try {
        showNotification('Training model... This may take a minute.', 'info');
        progressText.textContent = 'Preparing data...';
        
        const response = await fetch(`${API_BASE}/train`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
        });
        
        const data = await response.json();
        
        clearInterval(progressInterval);
        state.trainingProgress = 100;
        
        if (!response.ok) {
            throw new Error(data.error || 'Training failed');
        }
        
        // Update UI with results
        progressText.textContent = '✓ Training complete!';
        
        const modelName = document.getElementById('modelName');
        const accuracy = document.getElementById('accuracy');
        
        modelName.textContent = `Best Model: ${data.model_name}`;
        accuracy.textContent = `Accuracy: ${data.accuracy}`;
        
        trainingResult.classList.remove('hidden');
        updateModelStatus(true);
        
        showNotification(`Model trained successfully! Accuracy: ${data.accuracy}`, 'success');
        
    } catch (error) {
        clearInterval(progressInterval);
        console.error('Error:', error);
        showNotification(`Error: ${error.message}`, 'error');
        trainingProgress.classList.add('hidden');
        
    } finally {
        state.isTraining = false;
        trainBtn.disabled = false;
        
        // Keep progress visible for a moment
        setTimeout(() => {
            trainingProgress.classList.add('hidden');
        }, 2000);
    }
}

/**
 * Predict house price based on input features
 */
async function predictPrice() {
    if (!state.modelTrained) {
        showNotification('Please train the model first!', 'warning');
        return;
    }
    
    try {
        // Get form values - read fresh values every time
        const lotArea = parseFloat(document.getElementById('lotArea').value);
        const yearBuilt = parseFloat(document.getElementById('yearBuilt').value);
        const livArea = parseFloat(document.getElementById('livArea').value);
        const bedrooms = parseFloat(document.getElementById('bedrooms').value) || 0;
        const bathrooms = parseFloat(document.getElementById('bathrooms').value) || 0;
        const garageArea = parseFloat(document.getElementById('garageArea').value) || 0;
        
        // Validate inputs - must be numbers
        if (isNaN(lotArea) || isNaN(yearBuilt) || isNaN(livArea)) {
            showNotification('Please enter valid numbers for Lot Area, Year Built, and Living Area', 'warning');
            return;
        }
        
        // Validate positive values
        if (lotArea <= 0 || yearBuilt <= 0 || livArea <= 0) {
            showNotification('Lot Area, Year Built, and Living Area must be greater than 0', 'warning');
            return;
        }
        
        // Create feature object for API
        const features = {
            'LotArea': lotArea,
            'YearBuilt': yearBuilt,
            'GrLivArea': livArea,
            'BedroomAbvGr': bedrooms,
            'FullBath': bathrooms,
            'GarageArea': garageArea,
        };
        
        console.log('🔮 Making prediction with features:', features);
        
        // Get button and result elements
        const predictBtn = document.querySelector('.prediction-form .btn-secondary');
        const predictionResult = document.getElementById('predictionResult');
        
        if (!predictBtn || !predictionResult) {
            showNotification('UI elements not found', 'error');
            return;
        }
        
        // Disable button and show loading
        predictBtn.disabled = true;
        predictBtn.textContent = 'Predicting...';
        showNotification('Calculating prediction...', 'info');
        
        // Make API request
        console.log('📡 Sending prediction request...');
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ features }),
        });
        
        const data = await response.json();
        console.log('📥 Response received:', data);
        
        if (!response.ok) {
            throw new Error(data.error || `Prediction failed (${response.status})`);
        }
        
        if (!data.success || data.price_value === undefined) {
            throw new Error('Invalid response data from server');
        }
        
        // Update the price display - fresh update
        const predictedPriceEl = document.getElementById('predictedPrice');
        
        // Force element to be visible
        predictionResult.classList.remove('hidden');
        
        // Update price text
        const formattedPrice = formatINR(data.price_value);
        predictedPriceEl.textContent = formattedPrice;
        
        // Reset animation by removing and re-adding
        predictionResult.style.animation = 'none';
        void predictionResult.offsetHeight; // Trigger reflow
        predictionResult.style.animation = 'slideInUp 0.5s ease-out';
        
        showNotification('✅ Prediction complete!', 'success');
        console.log('✅ Prediction successful! Price:', formattedPrice);
        
    } catch (error) {
        console.error('❌ Prediction error:', error);
        showNotification(`Error: ${error.message}`, 'error');
        
    } finally {
        // Always re-enable button
        const predictBtn = document.querySelector('.prediction-form .btn-secondary');
        if (predictBtn) {
            predictBtn.disabled = false;
            predictBtn.textContent = 'Predict Price';
        }
    }
}

/**
 * Reset prediction form and result
 */
function resetPrediction() {
    console.log('🔄 Resetting prediction form...');
    
    // Reset form inputs to default values
    document.getElementById('lotArea').value = 8450;
    document.getElementById('yearBuilt').value = 2003;
    document.getElementById('livArea').value = 1710;
    document.getElementById('bedrooms').value = 3;
    document.getElementById('bathrooms').value = 2;
    document.getElementById('garageArea').value = 548;
    
    // Hide prediction result
    const predictionResult = document.getElementById('predictionResult');
    predictionResult.classList.add('hidden');
    
    // Reset price value
    const predictedPriceEl = document.getElementById('predictedPrice');
    predictedPriceEl.textContent = '₹0.00';
    
    showNotification('✅ Form reset! Ready for new prediction.', 'info');
    console.log('✅ Form reset complete');
}

/**
 * Check if model is already trained on page load
 */
async function checkModelStatus() {
    // Don't auto-check model status on load
    // Always start as "Not Trained" unless model.pkl exists
    const response = await fetch(`${API_BASE}/model-info`);
    const data = await response.json();
    
    if (data.trained) {
        updateModelStatus(true);
    } else {
        updateModelStatus(false);
    }
}

// Chart functions removed - analytics now text-only

/**
 * Format large numbers to INR format
 */
function formatINRValue(value) {
    if (value >= 10000000) {
        return (value / 10000000).toFixed(1) + 'Cr';
    } else if (value >= 100000) {
        return (value / 100000).toFixed(1) + 'L';
    }
    return value.toFixed(0);
}

/**
 * Load and display analytics data (text-only)
 */
async function loadAnalytics() {
    try {
        const response = await fetch(`${API_BASE}/analytics`);
        const data = await response.json();
        
        if (response.ok && data.average_price > 0) {
            state.analyticsData = data;
            console.log('Analytics loaded:', data);
            
            // Update analytics display with text information
            document.getElementById('totalRecords').textContent = data.total_records || '-';
            document.getElementById('avgPrice').textContent = formatINR(data.average_price);
            document.getElementById('medianPrice').textContent = formatINR(data.median_price);
        } else {
            console.log('Analytics data not ready or empty:', data);
        }
    } catch (error) {
        console.log('Analytics data error:', error);
    }
}

/**
 * Initialize price distribution chart
 */
document.addEventListener('DOMContentLoaded', () => {
    console.log('House Price Predictor - Application Loaded');
    
    // Always start as "Not Trained"
    updateModelStatus(false);
    
    // Load analytics data
    loadAnalytics();
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + T to train
        if ((e.ctrlKey || e.metaKey) && e.key === 't') {
            e.preventDefault();
            trainModel();
        }
        // Ctrl/Cmd + P to predict
        if ((e.ctrlKey || e.metaKey) && e.key === 'p') {
            e.preventDefault();
            predictPrice();
        }
    });
    
    // Handle Enter key in form
    document.getElementById('predictionForm').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            predictPrice();
        }
    });
    
    // Show initial message
    showNotification('Welcome! Start by training the model.', 'info');
});

/**
 * Utility: Format number as currency
 */
function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
    }).format(value);
}

/**
 * Utility: Validate email (if needed for future features)
 */
function validateEmail(email) {
    const regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return regex.test(email);
}

/**
 * Utility: Debounce function for performance
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Error boundary for unhandled errors
 */
window.addEventListener('error', (event) => {
    console.error('Unhandled error:', event.error);
    showNotification('An unexpected error occurred. Check console for details.', 'error');
});

/**
 * Log application state
 */
function logState() {
    console.table({
        'Is Training': state.isTraining,
        'Model Trained': state.modelTrained,
        'Training Progress': `${state.trainingProgress}%`,
    });
}
