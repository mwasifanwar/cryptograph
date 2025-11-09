# models/anomaly_detector.py
import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
import joblib

class AnomalyDetector:
    def __init__(self, method='isolation_forest'):
        self.method = method
        self.model = None
        self.is_trained = False
        
    def fit(self, embeddings):
        if self.method == 'isolation_forest':
            self.model = IsolationForest(contamination=0.1, random_state=42)
        elif self.method == 'one_class_svm':
            self.model = OneClassSVM(nu=0.1, kernel='rbf')
        elif self.method == 'dbscan':
            self.model = DBSCAN(eps=0.5, min_samples=5)
            
        if self.method != 'dbscan':
            self.model.fit(embeddings)
            self.is_trained = True
            
    def predict(self, embeddings):
        if not self.is_trained and self.method != 'dbscan':
            raise ValueError("Model not trained. Call fit() first.")
            
        if self.method == 'isolation_forest':
            scores = self.model.decision_function(embeddings)
            predictions = self.model.predict(embeddings)
            return predictions, scores
        elif self.method == 'one_class_svm':
            predictions = self.model.predict(embeddings)
            scores = self.model.decision_function(embeddings)
            return predictions, scores
        elif self.method == 'dbscan':
            predictions = self.model.fit_predict(embeddings)
            return predictions, np.zeros(len(embeddings))
            
    def save_model(self, filepath):
        if self.model is not None:
            joblib.dump(self.model, filepath)
            
    def load_model(self, filepath):
        self.model = joblib.load(filepath)
        self.is_trained = True

class DeepAnomalyDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DeepAnomalyDetector, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def compute_anomaly_score(self, x):
        reconstructed = self.forward(x)
        reconstruction_error = torch.mean((x - reconstructed) ** 2, dim=1)
        return reconstruction_error.detach().cpu().numpy()