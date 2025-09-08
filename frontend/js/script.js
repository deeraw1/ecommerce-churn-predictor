// ===== CONFIGURATION =====
const API_BASE = 'http://127.0.0.1:8000';
const APP_NAME = 'ChurnGuard AI';

// ===== DOM ELEMENTS =====
const elements = {
    apiStatus: document.getElementById('apiStatus'),
    connectionHelp: document.getElementById('connectionHelp'),
    singlePredictionForm: document.getElementById('singlePredictionForm'),
    singleSpinner: document.getElementById('singleSpinner'),
    singleSubmitText: document.getElementById('singleSubmitText'),
    singlePredictionResult: document.getElementById('singlePredictionResult'),
    noSinglePrediction: document.getElementById('noSinglePrediction'),
    batchPredictionForm: document.getElementById('batchPredictionForm'),
    batchSpinner: document.getElementById('batchSpinner'),
    batchSubmitText: document.getElementById('batchSubmitText'),
    batchResults: document.getElementById('batchResults'),
    noBatchResults: document.getElementById('noBatchResults'),
    historyContent: document.getElementById('historyContent'),
    refreshHistory: document.getElementById('refreshHistory'),
    totalPredictions: document.getElementById('totalPredictions'),
    churnRate: document.getElementById('churnRate'),
    highRisk: document.getElementById('highRisk')
};

// ===== INITIALIZATION =====
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    setDocumentTitle();
    setupEventListeners();
    checkApiHealth();
    loadPredictionHistory();
    loadGlobalStats();
}

function setDocumentTitle() {
    document.title = APP_NAME;
}

function setupEventListeners() {
    // Form submissions
    if (elements.singlePredictionForm) {
        elements.singlePredictionForm.addEventListener('submit', handleSinglePrediction);
    }
    
    if (elements.batchPredictionForm) {
        elements.batchPredictionForm.addEventListener('submit', handleBatchPrediction);
    }
    
    // History refresh
    if (elements.refreshHistory) {
        elements.refreshHistory.addEventListener('click', refreshHistory);
    }
    
    // Tab changes
    const tabEl = document.querySelector('a[data-bs-toggle="tab"]');
    if (tabEl) {
        tabEl.addEventListener('shown.bs.tab', function (e) {
            if (e.target.getAttribute('href') === '#history') {
                loadPredictionHistory();
            }
        });
    }
}

// ===== API HEALTH =====
async function checkApiHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        
        if (response.status === 404) {
            await checkApiViaHistory();
            return;
        }
        
        if (!response.ok) throw new Error(`HTTP error: ${response.status}`);
        
        const data = await response.json();
        updateApiStatus(data.status === 'healthy' ? 'healthy' : 'error', data);
        
    } catch (error) {
        await checkApiViaHistory();
    }
}

async function checkApiViaHistory() {
    try {
        const response = await fetch(`${API_BASE}/predictions/history?limit=1`);
        if (response.ok) {
            updateApiStatus('healthy', { message: 'API connected via history endpoint' });
        } else {
            throw new Error('All endpoints failed');
        }
    } catch (error) {
        updateApiStatus('error', { message: 'API unreachable' });
    }
}

function updateApiStatus(status, data) {
    const statusMap = {
        healthy: { text: 'API: Healthy', className: 'status-healthy', showHelp: false },
        error: { text: 'API: Error', className: 'status-error', showHelp: true },
        unreachable: { text: 'API: Unreachable', className: 'status-error', showHelp: true }
    };
    
    const statusInfo = statusMap[status] || statusMap.unreachable;
    
    elements.apiStatus.textContent = statusInfo.text;
    elements.apiStatus.className = `api-status ${statusInfo.className}`;
    elements.connectionHelp.style.display = statusInfo.showHelp ? 'block' : 'none';
}

// ===== SINGLE PREDICTION =====
async function handleSinglePrediction(e) {
    e.preventDefault();
    
    const formData = getFormData();
    if (!formData) return;
    
    setLoadingState('single', true);
    
    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });
        
        if (!response.ok) throw new Error(`API error: ${response.status}`);
        
        const data = await response.json();
        displaySinglePrediction(data);
        loadPredictionHistory();
        loadGlobalStats();
        
    } catch (error) {
        showError('Prediction failed: ' + error.message);
        console.error('Prediction error:', error);
    } finally {
        setLoadingState('single', false);
    }
}

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
    elements.noSinglePrediction.style.display = 'none';
    elements.singlePredictionResult.style.display = 'block';
    
    document.getElementById('churnPrediction').textContent = 
        data.churn_prediction === 1 ? 'YES' : 'NO';
    document.getElementById('churnProbability').textContent = 
        (data.churn_probability * 100).toFixed(1) + '%';
    
    const predictionIcon = document.getElementById('predictionIcon');
    predictionIcon.innerHTML = data.churn_prediction === 1 ? 
        '<i class="fas fa-exclamation-triangle text-danger fa-3x"></i>' :
        '<i class="fas fa-check-circle text-success fa-3x"></i>';
    
    const riskLevel = document.getElementById('riskLevel');
    riskLevel.textContent = `Risk Level: ${data.risk_level.toUpperCase()}`;
    riskLevel.className = `risk-${data.risk_level} fs-5`;
    
    displayInsights(data.insights);
}

function displayInsights(insights) {
    const insightsList = document.getElementById('insightsList');
    insightsList.innerHTML = '';
    
    insights.forEach(insight => {
        const li = document.createElement('li');
        li.className = 'list-group-item insight-card';
        li.innerHTML = `
            <div class="d-flex align-items-center">
                <i class="fas fa-lightbulb text-warning me-3"></i>
                <span>${insight}</span>
            </div>
        `;
        insightsList.appendChild(li);
    });
}

// ===== BATCH PREDICTION =====
async function handleBatchPrediction(e) {
    e.preventDefault();
    
    const file = document.getElementById('batchFile').files[0];
    if (!file) {
        showError('Please select a file first');
        return;
    }
    
    setLoadingState('batch', true);
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        const endpoint = file.name.endsWith('.csv') ? 
            '/predict/upload/csv' : '/predict/upload/excel';
        
        const response = await fetch(`${API_BASE}${endpoint}`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) throw new Error(`API error: ${response.status}`);
        
        const data = await response.json();
        displayBatchResults(data);
        loadPredictionHistory();
        loadGlobalStats();
        
    } catch (error) {
        showError('Batch processing failed: ' + error.message);
        console.error('Batch processing error:', error);
    } finally {
        setLoadingState('batch', false);
    }
}

function displayBatchResults(data) {
    elements.noBatchResults.style.display = 'none';
    
    let html = `
        <div class="alert alert-success animated">
            <h6><i class="fas fa-check-circle"></i> Processing Complete</h6>
            <p class="mb-1">File: ${data.filename}</p>
            <p class="mb-1">Processed: ${data.total_customers} customers</p>
            <p class="mb-1">Saved to database: ${data.saved_to_db} records</p>
        </div>
        
        <div class="alert alert-info animated">
            <h6><i class="fas fa-chart-bar"></i> Summary</h6>
            <p class="mb-1">Churn Count: <strong>${data.summary.churn_count}</strong></p>
            <p class="mb-1">Retention Count: <strong>${data.summary.retention_count}</strong></p>
            <p class="mb-1">Churn Rate: <strong>${(data.summary.churn_rate * 100).toFixed(1)}%</strong></p>
            <p class="mb-0">Avg Probability: <strong>${(data.summary.average_probability * 100).toFixed(1)}%</strong></p>
        </div>
    `;
    
    if (data.data_quality?.missing_columns?.length > 0) {
        html += `
            <div class="alert alert-warning animated">
                <h6><i class="fas fa-exclamation-triangle"></i> Data Quality Note</h6>
                <p>Missing columns: ${data.data_quality.missing_columns.join(', ')}</p>
                <p class="mb-0">Default values were used for these columns</p>
            </div>
        `;
    }
    
    if (data.predictions?.length > 0) {
        html += `<h6 class="mt-4">Sample Predictions (first ${data.predictions.length}):</h6>`;
        
        data.predictions.forEach(prediction => {
            html += createPredictionCard(prediction);
        });
        
        if (data.total_customers > data.predictions.length) {
            html += `<p class="small text-muted mt-3">${data.note}</p>`;
        }
    }
    
    elements.batchResults.innerHTML = html;
}

function createPredictionCard(prediction) {
    return `
        <div class="card mb-3 animated">
            <div class="card-body">
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <div>
                        <strong class="text-primary">Row ${prediction.row_id}</strong>
                        <span class="badge ${getRiskBadgeClass(prediction.risk_level)} ms-2">
                            ${prediction.risk_level.toUpperCase()}
                        </span>
                    </div>
                    <div>
                        <span class="me-2">Churn: <strong>${prediction.churn_prediction ? 'YES' : 'NO'}</strong></span>
                        <span class="text-muted">${(prediction.churn_probability * 100).toFixed(1)}%</span>
                    </div>
                </div>
                ${prediction.insights?.length > 0 ? `
                    <div class="mt-2">
                        <small class="text-muted">Key insights:</small>
                        <ul class="mb-0 ps-3">
                            ${prediction.insights.slice(0, 2).map(insight => 
                                `<li class="small">${insight}</li>`).join('')}
                        </ul>
                    </div>
                ` : ''}
            </div>
        </div>
    `;
}

function getRiskBadgeClass(riskLevel) {
    const riskClasses = {
        high: 'bg-danger',
        medium: 'bg-warning',
        low: 'bg-success'
    };
    return riskClasses[riskLevel] || 'bg-secondary';
}

// ===== HISTORY MANAGEMENT =====
async function loadPredictionHistory() {
    try {
        showHistoryLoading();
        
        const response = await fetch(`${API_BASE}/predictions/history?limit=20`);
        if (!response.ok) throw new Error(`API error: ${response.status}`);
        
        const data = await response.json();
        displayPredictionHistory(data);
        
    } catch (error) {
        elements.historyContent.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle"></i> Failed to load history: ${error.message}
            </div>
        `;
        console.error('History load error:', error);
    }
}

function showHistoryLoading() {
    elements.historyContent.innerHTML = `
        <div class="text-center py-4">
            <div class="spinner-border text-primary mb-3"></div>
            <p class="text-muted">Loading prediction history...</p>
        </div>
    `;
}

function displayPredictionHistory(data) {
    if (!data.count || data.count === 0) {
        elements.historyContent.innerHTML = `
            <div class="text-center py-5 text-muted">
                <i class="fas fa-history fa-3x mb-3"></i>
                <p>No prediction history found</p>
                <small>Make some predictions to see them here</small>
            </div>
        `;
        return;
    }
    
    let html = `<div class="history-stats mb-3">
        <small class="text-muted">Showing ${data.predictions.length} of ${data.count} predictions</small>
    </div>`;
    
    data.predictions.forEach(prediction => {
        html += createHistoryItem(prediction);
    });
    
    elements.historyContent.innerHTML = html;
}

function createHistoryItem(prediction) {
    const date = new Date(prediction.timestamp).toLocaleString();
    const riskLevel = prediction.churn_probability > 0.17 ? 'high' : 
                     prediction.churn_probability > 0.12 ? 'medium' : 'low';
    
    return `
        <div class="history-item animated">
            <div class="d-flex justify-content-between align-items-start">
                <div class="flex-grow-1">
                    <h6 class="mb-1 text-primary">Prediction on ${date}</h6>
                    <p class="mb-1 small">
                        <span class="me-2">Tenure: ${prediction.tenure}mo</span>
                        <span class="me-2">Orders: ${prediction.ordercount}</span>
                        <span>Satisfaction: ${prediction.satisfactionscore}/5</span>
                    </p>
                    <p class="mb-0">
                        <span class="badge ${getRiskBadgeClass(riskLevel)}">
                            Churn: ${prediction.churn_prediction ? 'YES' : 'NO'} 
                            (${(prediction.churn_probability * 100).toFixed(1)}%)
                        </span>
                    </p>
                </div>
            </div>
        </div>
    `;
}

function refreshHistory() {
    loadPredictionHistory();
}

// ===== GLOBAL STATS =====
async function loadGlobalStats() {
    try {
        const response = await fetch(`${API_BASE}/predictions/history?limit=1000`);
        if (!response.ok) return;
        
        const data = await response.json();
        updateGlobalStats(data.predictions);
        
    } catch (error) {
        console.error('Error loading global stats:', error);
    }
}

function updateGlobalStats(predictions) {
    if (!predictions || predictions.length === 0) return;
    
    const total = predictions.length;
    const churnCount = predictions.filter(p => p.churn_prediction === 1).length;
    const highRiskCount = predictions.filter(p => p.churn_probability > 0.17).length;
    const churnRate = (churnCount / total) * 100;
    
    if (elements.totalPredictions) {
        elements.totalPredictions.textContent = total.toLocaleString();
    }
    if (elements.churnRate) {
        elements.churnRate.textContent = churnRate.toFixed(1) + '%';
    }
    if (elements.highRisk) {
        elements.highRisk.textContent = highRiskCount.toLocaleString();
    }
}

// ===== UTILITY FUNCTIONS =====
function setLoadingState(type, isLoading) {
    const elementsMap = {
        single: {
            spinner: elements.singleSpinner,
            text: elements.singleSubmitText,
            defaultText: 'Predict Churn'
        },
        batch: {
            spinner: elements.batchSpinner,
            text: elements.batchSubmitText,
            defaultText: 'Process Batch Prediction'
        }
    };
    
    const { spinner, text, defaultText } = elementsMap[type];
    
    if (isLoading) {
        spinner.style.display = 'inline-block';
        text.textContent = type === 'single' ? 'Predicting...' : 'Processing...';
    } else {
        spinner.style.display = 'none';
        text.textContent = defaultText;
    }
}

function showError(message) {
    // You could use a toast notification library here
    alert(message);
}

// ===== EXPORT FUNCTIONS FOR GLOBAL ACCESS =====
window.ChurnGuard = {
    checkApiHealth,
    handleSinglePrediction,
    handleBatchPrediction,
    loadPredictionHistory,
    refreshHistory,
    loadGlobalStats
};