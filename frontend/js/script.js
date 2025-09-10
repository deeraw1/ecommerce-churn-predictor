// API base URL - point to your FastAPI server
const API_BASE = '';

// DOM elements
const apiStatus = document.getElementById('apiStatus');
const connectionHelp = document.getElementById('connectionHelp');
const singlePredictionForm = document.getElementById('singlePredictionForm');
const singleSpinner = document.getElementById('singleSpinner');
const singleSubmitText = document.getElementById('singleSubmitText');
const singlePredictionResult = document.getElementById('singlePredictionResult');
const noSinglePrediction = document.getElementById('noSinglePrediction');
const batchPredictionForm = document.getElementById('batchPredictionForm');
const batchSpinner = document.getElementById('batchSpinner');
const batchSubmitText = document.getElementById('batchSubmitText');
const batchResults = document.getElementById('batchResults');
const noBatchResults = document.getElementById('noBatchResults');
const historyContent = document.getElementById('historyContent');
const refreshHistory = document.getElementById('refreshHistory');
const batchFile = document.getElementById('batchFile');
const fileInfo = document.getElementById('fileInfo');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

// Initialize application
function initializeApp() {
    checkApiHealth();
    loadPredictionHistory();
    setupEventListeners();
    setupFileUpload();
}

// Setup event listeners
function setupEventListeners() {
    if (singlePredictionForm) {
        singlePredictionForm.addEventListener('submit', function(e) {
            e.preventDefault();
            handleSinglePrediction();
        });
    }
    
    if (batchPredictionForm) {
        batchPredictionForm.addEventListener('submit', function(e) {
            e.preventDefault();
            handleBatchPrediction();
        });
    }
    
    if (refreshHistory) {
        refreshHistory.addEventListener('click', function() {
            showLoadingHistory();
            loadPredictionHistory();
        });
    }
}

// Setup file upload functionality
function setupFileUpload() {
    if (batchFile) {
        batchFile.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                updateFileInfo(file);
            }
        });
        
        // Drag and drop functionality
        const dropzone = batchPredictionForm;
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, unhighlight, false);
        });
        
        function highlight() {
            dropzone.classList.add('highlight');
        }
        
        function unhighlight() {
            dropzone.classList.remove('highlight');
        }
        
        dropzone.addEventListener('drop', handleDrop, false);
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            batchFile.files = files;
            updateFileInfo(files[0]);
        }
    }
}

// Update file information display
function updateFileInfo(file) {
    if (file && fileInfo) {
        const fileSize = (file.size / 1024 / 1024).toFixed(2);
        fileInfo.innerHTML = `
            <div class="alert alert-info mt-3">
                <i class="fas fa-file me-2"></i>
                <strong>${file.name}</strong> (${fileSize} MB)
                <br>
                <small>Type: ${file.type || 'Unknown'}</small>
            </div>
        `;
    }
}

// Check API health status
async function checkApiHealth() {
    try {
        const response = await fetch(API_BASE + '/health');
        
        if (response.status === 404) {
            // Try history endpoint as fallback
            const historyResponse = await fetch(API_BASE + '/predictions/history?limit=1');
            if (historyResponse.ok) {
                updateApiStatus('healthy', 'API: Healthy (no /health endpoint)');
                return;
            }
            throw new Error('Health endpoint not found');
        }
        
        if (!response.ok) throw new Error(`HTTP error: ${response.status}`);
        
        const data = await response.json();
        
        if (data.status === 'healthy') {
            updateApiStatus('healthy', 'API: Healthy');
        } else {
            updateApiStatus('error', 'API: Error');
        }
    } catch (error) {
        // Fallback check
        try {
            const testResponse = await fetch(API_BASE + '/predictions/history?limit=1');
            if (testResponse.ok) {
                updateApiStatus('healthy', 'API: Connected');
            } else {
                throw new Error('All endpoints failed');
            }
        } catch (fallbackError) {
            updateApiStatus('error', 'API: Unreachable');
            console.error('API health check failed:', error);
        }
    }
}

// Update API status display
function updateApiStatus(status, message) {
    if (!apiStatus) return;
    
    apiStatus.className = 'api-status status-' + status;
    apiStatus.innerHTML = `<i class="fas fa-${getStatusIcon(status)}"></i> ${message}`;
    
    if (connectionHelp) {
        connectionHelp.style.display = status === 'error' ? 'block' : 'none';
    }
}

// Get status icon based on status
function getStatusIcon(status) {
    const icons = {
        healthy: 'check-circle',
        warning: 'exclamation-triangle',
        error: 'times-circle'
    };
    return icons[status] || 'question-circle';
}

// Handle single prediction form submission
async function handleSinglePrediction() {
    showLoading(singleSpinner, singleSubmitText, 'Predicting...');
    
    try {
        const formData = getFormData();
        const response = await fetch(API_BASE + '/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });
        
        if (!response.ok) throw new Error(`API error: ${response.status}`);
        
        const data = await response.json();
        displaySinglePrediction(data);
        loadPredictionHistory();
        
    } catch (error) {
        showError('Prediction failed: ' + error.message);
        console.error('Prediction error:', error);
    } finally {
        hideLoading(singleSpinner, singleSubmitText, 'Predict Churn');
    }
}

// Get form data from single prediction form
function getFormData() {
    return {
        tenure: parseInt(document.getElementById('tenure').value),
        numberofaddress: parseInt(document.getElementById('numberofaddress').value),
        cashbackamount: parseFloat(document.getElementById('cashbackamount').value),
        daysincelastorder: parseInt(document.getElementById('daysincelastorder').value),
        ordercount: parseInt(document.getElementById('ordercount').value),
        satisfactionscore: parseInt(document.getElementById('satisfactionscore').value)
    };
}

function displaySinglePrediction(data) {
    noSinglePrediction.style.display = 'none';
    singlePredictionResult.style.display = 'block';
    
    // Set prediction text and icon
    document.getElementById('churnPrediction').textContent = 
        data.churn_prediction === 1 ? 'YES' : 'NO';
    document.getElementById('churnProbability').textContent = 
        (data.churn_probability * 100).toFixed(1) + '%';
    
    // Set prediction icon
    const predictionIcon = document.getElementById('predictionIcon');
    predictionIcon.innerHTML = data.churn_prediction === 1 ? 
        '<i class="fas fa-exclamation-triangle text-danger"></i>' :
        '<i class="fas fa-check-circle text-success"></i>';
    
    // Set risk level
    const riskLevel = document.getElementById('riskLevel');
    riskLevel.textContent = 'Risk Level: ' + data.risk_level.toUpperCase();
    riskLevel.className = 'risk-badge risk-' + data.risk_level;
    
    // Display insights
    displayInsights(data.insights);
    
    // Add download button
    addDownloadButtonToSingleResult(data);
    
    // Add animation
    singlePredictionResult.classList.add('fade-in');
}

// Add this new function for single result download button
function addDownloadButtonToSingleResult(data) {
    // Remove existing download button if any
    const existingButton = document.getElementById('downloadSingleBtn');
    if (existingButton) existingButton.remove();
    
    const downloadButton = handleSingleDownload(data);
    if (downloadButton) {
        downloadButton.id = 'downloadSingleBtn';
        downloadButton.className = 'btn btn-success btn-gradient mt-3';
        
        const insightsContainer = document.getElementById('insightsList');
        insightsContainer.parentNode.insertBefore(downloadButton, insightsContainer.nextSibling);
    }
}

// Display insights list
function displayInsights(insights) {
    const insightsList = document.getElementById('insightsList');
    insightsList.innerHTML = '';
    
    insights.forEach(insight => {
        const insightElement = document.createElement('div');
        insightElement.className = 'insight-item';
        insightElement.innerHTML = `
            <i class="fas fa-arrow-right me-2 text-primary"></i>
            ${insight}
        `;
        insightsList.appendChild(insightElement);
    });
}

// Handle batch prediction form submission
async function handleBatchPrediction() {
    const file = batchFile.files[0];
    if (!file) {
        showError('Please select a file first');
        return;
    }
    
    showLoading(batchSpinner, batchSubmitText, 'Processing...');
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        const endpoint = file.name.endsWith('.csv') ? 
            '/predict/upload/csv' : '/predict/upload/excel';
        
        const response = await fetch(API_BASE + endpoint, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) throw new Error(`API error: ${response.status}`);
        
        const data = await response.json();
        displayBatchResults(data);
        loadPredictionHistory();
        
    } catch (error) {
        showError('Batch processing failed: ' + error.message);
        console.error('Batch processing error:', error);
    } finally {
        hideLoading(batchSpinner, batchSubmitText, 'Process Batch Prediction');
    }
}

function displayBatchResults(data) {
    noBatchResults.style.display = 'none';
    
    let html = `
        <div class="alert alert-success fade-in">
            <h6><i class="fas fa-check-circle"></i> Processing Complete</h6>
            <p class="mb-1"><strong>File:</strong> ${data.filename}</p>
            <p class="mb-1"><strong>Processed:</strong> ${data.total_customers} customers</p>
            <p class="mb-0"><strong>Saved:</strong> ${data.saved_to_db} records</p>
        </div>
    `;
    
    // Add download button at the top
    if (data.predictions?.length > 0) {
        const downloadButton = handleBatchDownload(data);
        if (downloadButton) {
            html += `<div class="d-flex justify-content-end mb-3">${downloadButton.outerHTML}</div>`;
        }
    }
    
    html += `
        <div class="alert alert-info fade-in">
            <h6><i class="fas fa-chart-bar"></i> Summary</h6>
            <div class="row">
                <div class="col-6">
                    <p class="mb-1">Churn: ${data.summary.churn_count}</p>
                    <p class="mb-0">Retention: ${data.summary.retention_count}</p>
                </div>
                <div class="col-6">
                    <p class="mb-1">Churn Rate: ${(data.summary.churn_rate * 100).toFixed(1)}%</p>
                    <p class="mb-0">Avg Probability: ${(data.summary.average_probability * 100).toFixed(1)}%</p>
                </div>
            </div>
        </div>
    `;
    
    if (data.data_quality?.missing_columns?.length > 0) {
        html += `
            <div class="alert alert-warning fade-in">
                <h6><i class="fas fa-exclamation-triangle"></i> Data Quality</h6>
                <p>Missing columns: ${data.data_quality.missing_columns.join(', ')}</p>
                <p class="mb-0">Default values were used</p>
            </div>
        `;
    }
    
    if (data.predictions?.length > 0) {
        html += '<h6 class="section-title">Sample Predictions:</h6>';
        
        data.predictions.forEach(prediction => {
            html += createPredictionCard(prediction);
        });
        
        if (data.total_customers > data.predictions.length) {
            html += `<p class="small text-muted fade-in">${data.note}</p>`;
        }
        
        // Add download button at the bottom too
        const downloadButton = handleBatchDownload(data);
        if (downloadButton) {
            html += `<div class="text-center mt-4">${downloadButton.outerHTML}</div>`;
        }
    }
    
    batchResults.innerHTML = html;
}

// Create prediction card for batch results
function createPredictionCard(prediction) {
    const riskClass = prediction.risk_level === 'high' ? 'danger' : 
                     prediction.risk_level === 'medium' ? 'warning' : 'success';
    
    return `
        <div class="card mb-3 fade-in">
            <div class="card-body">
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <h6 class="mb-0">Row ${prediction.row_id}</h6>
                    <span class="badge bg-${riskClass}">
                        ${prediction.risk_level.toUpperCase()}
                    </span>
                </div>
                <div class="row">
                    <div class="col-6">
                        <small>Churn: ${prediction.churn_prediction ? 'YES' : 'NO'}</small>
                    </div>
                    <div class="col-6 text-end">
                        <small>${(prediction.churn_probability * 100).toFixed(1)}%</small>
                    </div>
                </div>
                ${prediction.insights?.length > 0 ? `
                    <div class="mt-2">
                        ${prediction.insights.slice(0, 2).map(insight => `
                            <div class="small text-muted">
                                <i class="fas fa-arrow-right me-1"></i>${insight}
                            </div>
                        `).join('')}
                    </div>
                ` : ''}
            </div>
        </div>
    `;
}

// Load prediction history
async function loadPredictionHistory() {
    try {
        const response = await fetch(API_BASE + '/predictions/history?limit=20');
        
        if (!response.ok) throw new Error(`API error: ${response.status}`);
        
        const data = await response.json();
        displayPredictionHistory(data);
        
    } catch (error) {
        historyContent.innerHTML = `
            <div class="alert alert-danger fade-in">
                <i class="fas fa-exclamation-triangle"></i> 
                Failed to load history: ${error.message}
            </div>
        `;
        console.error('History load error:', error);
    }
}

// Display prediction history
function displayPredictionHistory(data) {
    if (!data.count || data.count === 0) {
        historyContent.innerHTML = `
            <div class="empty-state fade-in">
                <i class="fas fa-history fa-3x mb-3"></i>
                <p>No prediction history found</p>
            </div>
        `;
        return;
    }
    
    let html = '';
    
    data.predictions.forEach(prediction => {
        html += createHistoryItem(prediction);
    });
    
    historyContent.innerHTML = html;
}

// Create history item
function createHistoryItem(prediction) {
    const date = new Date(prediction.timestamp).toLocaleString();
    const riskLevel = prediction.churn_probability > 0.17 ? 'high' : 
                     prediction.churn_probability > 0.12 ? 'medium' : 'low';
    const riskClass = riskLevel === 'high' ? 'danger' : 
                     riskLevel === 'medium' ? 'warning' : 'success';
    
    return `
        <div class="history-item fade-in">
            <div class="d-flex justify-content-between align-items-start">
                <div class="flex-grow-1">
                    <h6 class="mb-1">Prediction on ${date}</h6>
                    <div class="row">
                        <div class="col-6">
                            <small class="text-muted">
                                <i class="fas fa-calendar me-1"></i>${prediction.tenure}mo
                            </small>
                        </div>
                        <div class="col-6">
                            <small class="text-muted">
                                <i class="fas fa-shopping-cart me-1"></i>${prediction.ordercount} orders
                            </small>
                        </div>
                    </div>
                    <div class="mt-2">
                        <span class="badge bg-${riskClass}">
                            <i class="fas fa-${riskLevel === 'high' ? 'exclamation-triangle' : 'chart-line'} me-1"></i>
                            Churn: ${prediction.churn_prediction ? 'YES' : 'NO'} (${(prediction.churn_probability * 100).toFixed(1)}%)
                        </span>
                    </div>
                </div>
            </div>
        </div>
    `;
}

// Show loading state for history
function showLoadingHistory() {
    historyContent.innerHTML = `
        <div class="loading-state">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-3">Loading prediction history...</p>
        </div>
    `;
}

// Show loading state
function showLoading(spinner, textElement, message) {
    if (spinner) spinner.style.display = 'inline-block';
    if (textElement) textElement.textContent = message;
}

// Hide loading state
function hideLoading(spinner, textElement, defaultText) {
    if (spinner) spinner.style.display = 'none';
    if (textElement) textElement.textContent = defaultText;
}

// Add this function to create download buttons
function createDownloadButton(data, filename, buttonText) {
    const blob = new Blob([data], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    
    const button = document.createElement('a');
    button.href = url;
    button.download = filename;
    button.className = 'btn btn-success btn-sm me-2';
    button.innerHTML = `<i class="fas fa-download me-1"></i> ${buttonText}`;
    
    return button;
}

// Add this function to format CSV data
function formatPredictionsToCSV(predictions) {
    const headers = ['Row_ID', 'Churn_Prediction', 'Churn_Probability', 'Risk_Level', 'Insights'];
    let csv = headers.join(',') + '\n';
    
    predictions.forEach(pred => {
        const row = [
            pred.row_id,
            pred.churn_prediction ? 'Churned' : 'Retained',
            pred.churn_probability,
            pred.risk_level,
            pred.insights ? pred.insights.join('; ') : ''
        ].map(field => `"${field}"`).join(',');
        
        csv += row + '\n';
    });
    
    return csv;
}

// Add this function to handle batch download
function handleBatchDownload(data) {
    if (!data.predictions || data.predictions.length === 0) return null;
    
    const csvData = formatPredictionsToCSV(data.predictions);
    return createDownloadButton(csvData, `churn_predictions_${Date.now()}.csv`, 'Download Results');
}

// Add this function for single prediction download
function handleSingleDownload(predictionData) {
    const data = [{
        row_id: 1,
        churn_prediction: predictionData.churn_prediction,
        churn_probability: predictionData.churn_probability,
        risk_level: predictionData.risk_level,
        insights: predictionData.insights
    }];
    
    const csvData = formatPredictionsToCSV(data);
    return createDownloadButton(csvData, `churn_prediction_${Date.now()}.csv`, 'Download Result');
}

// Show error message
function showError(message) {
    // You can replace this with a toast notification system
    alert(message);
}

// Export functions for potential module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        initializeApp,
        checkApiHealth,
        handleSinglePrediction,
        handleBatchPrediction,
        loadPredictionHistory
    };
}