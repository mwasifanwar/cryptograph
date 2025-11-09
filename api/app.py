# api/app.py
from flask import Flask, request, jsonify, render_template
import os
import json
from werkzeug.utils import secure_filename
import pandas as pd

from data.blockchain_loader import BlockchainDataLoader
from data.graph_builder import GraphBuilder
from analysis.pattern_analyzer import PatternAnalyzer
from analysis.risk_assessor import RiskAssessor
from models.anomaly_detector import AnomalyDetector
from config.settings import config

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

blockchain_loader = BlockchainDataLoader(config.ETHEREUM_RPC_URL)
graph_builder = GraphBuilder()
pattern_analyzer = PatternAnalyzer()
risk_assessor = RiskAssessor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze/address', methods=['POST'])
def analyze_address():
    data = request.get_json()
    address = data.get('address')
    
    if not address:
        return jsonify({'error': 'Address required'}), 400
        
    try:
        transactions = blockchain_loader.get_address_transactions(address, limit=1000)
        if not transactions:
            return jsonify({'error': 'No transactions found'}), 404
            
        df = blockchain_loader.create_transaction_dataframe(transactions)
        graph = graph_builder.build_transaction_graph(df)
        
        patterns = pattern_analyzer.detect_money_laundering_patterns(graph)
        network_analysis = pattern_analyzer.analyze_transaction_network(graph)
        
        result = {
            'address': address,
            'transaction_count': len(transactions),
            'patterns_detected': len(patterns) > 0,
            'suspicious_patterns': patterns,
            'network_metrics': network_analysis
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze/batch', methods=['POST'])
def analyze_batch():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    try:
        df = pd.read_csv(file)
        results = []
        
        for address in df['address'].head(100):
            transactions = blockchain_loader.get_address_transactions(address, limit=500)
            if transactions:
                tx_df = blockchain_loader.create_transaction_dataframe(transactions)
                graph = graph_builder.build_transaction_graph(tx_df)
                patterns = pattern_analyzer.detect_money_laundering_patterns(graph)
                
                results.append({
                    'address': address,
                    'suspicious': len(patterns) > 0,
                    'pattern_count': len(patterns)
                })
                
        return jsonify({'results': results, 'total_analyzed': len(results)})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/detect/anomalies', methods=['POST'])
def detect_anomalies():
    data = request.get_json()
    transactions_data = data.get('transactions', [])
    
    try:
        df = pd.DataFrame(transactions_data)
        graph = graph_builder.build_transaction_graph(df)
        pyg_data = graph_builder.graph_to_pyg_data(graph)
        
        detector = AnomalyDetector(method='isolation_forest')
        node_features = pyg_data.x.numpy()
        detector.fit(node_features)
        
        predictions, scores = detector.predict(node_features)
        
        anomalies = []
        for i, (pred, score) in enumerate(zip(predictions, scores)):
            if pred == -1:
                node = list(graph.nodes())[i]
                anomalies.append({
                    'node': node,
                    'anomaly_score': float(score),
                    'risk_level': 'high' if abs(score) > 0.5 else 'medium'
                })
                
        return jsonify({
            'total_nodes': len(graph.nodes()),
            'anomalies_detected': len(anomalies),
            'anomalous_nodes': anomalies
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate/report', methods=['POST'])
def generate_report():
    data = request.get_json()
    addresses = data.get('addresses', [])
    
    try:
        all_transactions = []
        for address in addresses[:10]:
            transactions = blockchain_loader.get_address_transactions(address, limit=200)
            all_transactions.extend(transactions)
            
        df = blockchain_loader.create_transaction_dataframe(all_transactions)
        graph = graph_builder.build_transaction_graph(df)
        
        patterns = pattern_analyzer.detect_money_laundering_patterns(graph)
        suspicious_subgraphs = pattern_analyzer.generate_suspicious_subgraphs(graph)
        
        report = {
            'analysis_summary': {
                'total_addresses_analyzed': len(addresses),
                'total_transactions': len(all_transactions),
                'network_size': graph.number_of_nodes(),
                'suspicious_patterns_found': len(patterns)
            },
            'detailed_findings': patterns,
            'risk_assessment': {
                'high_risk_patterns': len([p for p in patterns if patterns[p]]),
                'recommended_actions': []
            }
        }
        
        if patterns['high_frequency_nodes']:
            report['risk_assessment']['recommended_actions'].append(
                "Investigate high-frequency trading addresses"
            )
            
        if patterns['cyclic_transactions']:
            report['risk_assessment']['recommended_actions'].append(
                "Review circular transaction patterns for potential money laundering"
            )
            
        return jsonify(report)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)