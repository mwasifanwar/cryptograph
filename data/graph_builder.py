# data/graph_builder.py
import networkx as nx
import pandas as pd
import numpy as np
from torch_geometric.data import Data
import torch

class GraphBuilder:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        
    def build_transaction_graph(self, transactions_df):
        G = nx.MultiDiGraph()
        
        for _, tx in transactions_df.iterrows():
            from_addr = tx['from']
            to_addr = tx['to']
            
            if from_addr not in G:
                G.add_node(from_addr, type='address')
            if to_addr not in G:
                G.add_node(to_addr, type='address')
                
            edge_features = {
                'value': float(tx['value_eth']),
                'gas_price': float(tx['gas_price_gwei']),
                'timestamp': float(tx['timestamp'].timestamp()),
                'gas_used': float(tx['gas_used']),
                'input_length': float(tx['input_length'])
            }
            
            G.add_edge(from_addr, to_addr, **edge_features, tx_hash=tx['hash'])
            
        return G

    def graph_to_pyg_data(self, graph, node_features=None):
        node_mapping = {node: i for i, node in enumerate(graph.nodes())}
        
        edge_index = []
        edge_attr = []
        
        for u, v, data in graph.edges(data=True):
            src = node_mapping[u]
            dst = node_mapping[v]
            edge_index.append([src, dst])
            
            features = [
                data.get('value', 0),
                data.get('gas_price', 0),
                data.get('timestamp', 0),
                data.get('gas_used', 0),
                data.get('input_length', 0)
            ]
            edge_attr.append(features)
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        if node_features is None:
            node_features = torch.ones(len(graph.nodes()), 10)
        else:
            node_features = torch.tensor(node_features, dtype=torch.float)
        
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

    def extract_subgraph(self, graph, center_node, radius=2):
        try:
            ego_graph = nx.ego_graph(graph, center_node, radius=radius, undirected=True)
            return ego_graph
        except:
            return None

    def calculate_graph_metrics(self, graph):
        metrics = {}
        
        try:
            metrics['num_nodes'] = graph.number_of_nodes()
            metrics['num_edges'] = graph.number_of_edges()
            metrics['density'] = nx.density(graph)
            metrics['average_degree'] = sum(dict(graph.degree()).values()) / metrics['num_nodes']
            
            if nx.is_strongly_connected(graph.to_undirected()):
                metrics['average_clustering'] = nx.average_clustering(graph.to_undirected())
            else:
                metrics['average_clustering'] = 0
                
        except Exception as e:
            metrics['error'] = str(e)
            
        return metrics