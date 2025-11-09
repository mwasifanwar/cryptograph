# main.py
from api.app import app
from config.settings import config
import os

if __name__ == '__main__':
    os.makedirs('trained_models', exist_ok=True)
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    print("Starting CryptoGraph Blockchain Anomaly Detection System...")
    print("Access the web interface at: http://localhost:5000")
    print("API endpoints available for programmatic access")
    
    app.run(debug=True, host='0.0.0.0', port=5000)