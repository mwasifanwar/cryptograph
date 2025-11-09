# features/feature_engineer.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import networkx as nx

class BlockchainFeatureEngineer:
    def __init__(self):
        self.scalers = {}
        
    def create_node_features(self, graph, node):
        features = []
        
        try:
            in_degree = graph.in_degree(node)
            out_degree = graph.out_degree(node)
            total_degree = in_degree + out_degree
            
            in_values = [data['value'] for _, _, data in graph.in_edges(node, data=True)]
            out_values = [data['value'] for _, _, data in graph.out_edges(node, data=True)]
            
            avg_in_value = np.mean(in_values) if in_values else 0
            avg_out_value = np.mean(out_values) if out_values else 0
            total_volume = sum(in_values) + sum(out_values)
            
            features.extend([
                in_degree,
                out_degree,
                total_degree,
                avg_in_value,
                avg_out_value,
                total_volume,
                len(in_values),
                len(out_values),
                np.std(in_values) if in_values else 0,
                np.std(out_values) if out_values else 0
            ])
            
        except Exception as e:
            features = [0] * 10
            
        return features

    def create_graph_features(self, graph):
        features = {}
        
        features['num_nodes'] = graph.number_of_nodes()
        features['num_edges'] = graph.number_of_edges()
        features['density'] = nx.density(graph)
        
        degrees = [deg for _, deg in graph.degree()]
        features['avg_degree'] = np.mean(degrees) if degrees else 0
        features['max_degree'] = np.max(degrees) if degrees else 0
        
        try:
            if nx.is_strongly_connected(graph.to_undirected()):
                features['avg_clustering'] = nx.average_clustering(graph.to_undirected())
            else:
                features['avg_clustering'] = 0
        except:
            features['avg_clustering'] = 0
            
        edge_values = [data['value'] for _, _, data in graph.edges(data=True)]
        features['total_volume'] = sum(edge_values) if edge_values else 0
        features['avg_transaction_value'] = np.mean(edge_values) if edge_values else 0
        
        return features

    def normalize_features(self, features, method='standard'):
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
            
        normalized = scaler.fit_transform(features)
        self.scalers[method] = scaler
        return normalized

    def extract_temporal_features(self, transactions_df):
        transactions_df['hour'] = transactions_df['timestamp'].dt.hour
        transactions_df['day_of_week'] = transactions_df['timestamp'].dt.dayofweek
        transactions_df['month'] = transactions_df['timestamp'].dt.month
        
        hourly_volume = transactions_df.groupby('hour')['value_eth'].sum()
        daily_pattern = transactions_df.groupby('day_of_week')['value_eth'].sum()
        
        return {
            'hourly_volume': hourly_volume.to_dict(),
            'daily_pattern': daily_pattern.to_dict(),
            'total_daily_volume': transactions_df['value_eth'].sum(),
            'avg_daily_volume': transactions_df['value_eth'].mean(),
            'volume_std': transactions_df['value_eth'].std()
        }

    def detect_behavioral_patterns(self, graph, node):
        patterns = {}
        
        try:
            neighbors = list(graph.neighbors(node))
            patterns['neighbor_count'] = len(neighbors)
            
            edge_data = list(graph.edges(node, data=True))
            values = [data['value'] for _, _, data in edge_data]
            patterns['transaction_count'] = len(values)
            patterns['total_value'] = sum(values)
            patterns['avg_value'] = np.mean(values) if values else 0
            patterns['value_std'] = np.std(values) if values else 0
            
            if len(values) > 1:
                patterns['value_cv'] = patterns['value_std'] / patterns['avg_value']
            else:
                patterns['value_cv'] = 0
                
        except Exception as e:
            patterns = {'error': str(e)}
            
        return patterns