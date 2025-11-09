# config/settings.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    ETHEREUM_RPC_URL = os.getenv('ETHEREUM_RPC_URL', 'https://mainnet.infura.io/v3/your-key')
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    HIDDEN_DIM = 128
    NUM_EPOCHS = 100
    MODEL_SAVE_PATH = "trained_models/blockchain_gnn.pth"
    
    ANOMALY_THRESHOLDS = {
        'low': 0.3,
        'medium': 0.6,
        'high': 0.8
    }
    
    TRANSACTION_FEATURES = [
        'value', 'gas', 'gas_price', 'timestamp', 'input_length'
    ]
    
    GRAPH_PARAMS = {
        'max_nodes': 10000,
        'feature_dim': 10,
        'edge_dim': 3
    }

config = Config()