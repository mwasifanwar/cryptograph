# analysis/risk_assessor.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

class RiskAssessor:
    def __init__(self):
        self.risk_model = None
        self.feature_importance = None
        
    def train_risk_model(self, features, labels):
        self.risk_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.risk_model.fit(features, labels)
        self.feature_importance = dict(zip(
            range(len(features.columns)), 
            self.risk_model.feature_importances_
        ))
        
    def assess_risk(self, features):
        if self.risk_model is None:
            raise ValueError("Risk model not trained. Call train_risk_model first.")
            
        probabilities = self.risk_model.predict_proba(features)
        predictions = self.risk_model.predict(features)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'risk_scores': probabilities[:, 1]
        }
        
    def calculate_composite_risk_score(self, graph_metrics, node_features, transaction_patterns):
        risk_factors = []
        
        if graph_metrics.get('density', 0) > 0.1:
            risk_factors.append(0.3)
            
        if node_features.get('degree', 0) > 100:
            risk_factors.append(0.4)
            
        if transaction_patterns.get('value_std', 0) > 1000:
            risk_factors.append(0.5)
            
        if len(risk_factors) == 0:
            return 0.1
            
        return np.mean(risk_factors)

    def generate_risk_report(self, graph, suspicious_nodes):
        report = {
            'total_nodes': graph.number_of_nodes(),
            'total_edges': graph.number_of_edges(),
            'suspicious_nodes_count': len(suspicious_nodes),
            'risk_distribution': {},
            'recommendations': []
        }
        
        risk_levels = {'low': 0, 'medium': 0, 'high': 0}
        for node, risk in suspicious_nodes:
            if risk < 0.3:
                risk_levels['low'] += 1
            elif risk < 0.7:
                risk_levels['medium'] += 1
            else:
                risk_levels['high'] += 1
                
        report['risk_distribution'] = risk_levels
        
        if risk_levels['high'] > 10:
            report['recommendations'].append("Immediate investigation required for high-risk addresses")
        if risk_levels['medium'] > 50:
            report['recommendations'].append("Enhanced monitoring recommended for medium-risk patterns")
            
        return report

    def save_model(self, filepath):
        if self.risk_model is not None:
            joblib.dump(self.risk_model, filepath)
            
    def load_model(self, filepath):
        self.risk_model = joblib.load(filepath)