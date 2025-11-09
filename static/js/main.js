// static/js/main.js
class CryptoGraphApp {
    constructor() {
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        document.getElementById('addressForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.analyzeAddress();
        });

        document.getElementById('batchForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.analyzeBatch();
        });

        document.getElementById('realTimeBtn').addEventListener('click', () => {
            this.startRealTimeMonitoring();
        });
    }

    async analyzeAddress() {
        const address = document.getElementById('addressInput').value;
        if (!address) {
            this.showError('Please enter an Ethereum address');
            return;
        }

        this.showLoading();

        try {
            const response = await fetch('/analyze/address', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ address: address })
            });

            const result = await response.json();
            
            if (response.ok) {
                this.displayAddressResults(result);
            } else {
                this.showError(result.error);
            }
        } catch (error) {
            this.showError('Analysis failed: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    async analyzeBatch() {
        const fileInput = document.getElementById('batchFile');
        const file = fileInput.files[0];
        
        if (!file) {
            this.showError('Please select a CSV file');
            return;
        }

        this.showLoading();

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/analyze/batch', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            
            if (response.ok) {
                this.displayBatchResults(result);
            } else {
                this.showError(result.error);
            }
        } catch (error) {
            this.showError('Batch analysis failed: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    displayAddressResults(result) {
        const resultsContainer = document.getElementById('results');
        
        let html = `
            <h2>Analysis Results for ${result.address}</h2>
            <div class="summary">
                <p><strong>Transaction Count:</strong> ${result.transaction_count}</p>
                <p><strong>Suspicious Patterns:</strong> ${result.patterns_detected ? 'Yes' : 'No'}</p>
            </div>
        `;

        if (result.patterns_detected) {
            html += `<h3>Detected Patterns:</h3>`;
            for (const [patternType, patterns] of Object.entries(result.suspicious_patterns)) {
                if (patterns && patterns.length > 0) {
                    html += `
                        <div class="pattern-item">
                            <h4>${this.formatPatternName(patternType)}</h4>
                            <p>Found ${patterns.length} instances</p>
                        </div>
                    `;
                }
            }
        }

        resultsContainer.innerHTML = html;
        resultsContainer.style.display = 'block';
    }

    displayBatchResults(result) {
        const resultsContainer = document.getElementById('results');
        
        let html = `
            <h2>Batch Analysis Results</h2>
            <div class="summary">
                <p><strong>Total Analyzed:</strong> ${result.total_analyzed}</p>
                <p><strong>Suspicious Addresses:</strong> ${result.results.filter(r => r.suspicious).length}</p>
            </div>
        `;

        const suspicious = result.results.filter(r => r.suspicious);
        if (suspicious.length > 0) {
            html += `<h3>Suspicious Addresses:</h3>`;
            suspicious.forEach(addr => {
                html += `
                    <div class="pattern-item">
                        <p><strong>Address:</strong> ${addr.address}</p>
                        <p><strong>Patterns Detected:</strong> ${addr.pattern_count}</p>
                    </div>
                `;
            });
        }

        resultsContainer.innerHTML = html;
        resultsContainer.style.display = 'block';
    }

    formatPatternName(patternType) {
        return patternType.split('_').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    }

    showLoading() {
        const resultsContainer = document.getElementById('results');
        resultsContainer.innerHTML = `
            <div class="loading">
                <div class="spinner"></div>
                <p>Analyzing blockchain data... This may take a few moments.</p>
            </div>
        `;
        resultsContainer.style.display = 'block';
    }

    hideLoading() {
        // Loading is replaced by results
    }

    showError(message) {
        const resultsContainer = document.getElementById('results');
        resultsContainer.innerHTML = `
            <div class="error">
                <h3>Error</h3>
                <p>${message}</p>
            </div>
        `;
        resultsContainer.style.display = 'block';
    }

    startRealTimeMonitoring() {
        alert('Real-time monitoring feature coming soon!');
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new CryptoGraphApp();
});