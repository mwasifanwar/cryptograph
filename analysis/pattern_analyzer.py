# analysis/pattern_analyzer.py
import networkx as nx
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

class PatternAnalyzer:
    def __init__(self):
        self.patterns = defaultdict(list)
        
    def detect_money_laundering_patterns(self, graph):
        patterns = {}
        
        patterns['high_frequency_nodes'] = self._find_high_frequency_nodes(graph)
        patterns['high_value_clusters'] = self._find_high_value_clusters(graph)
        patterns['cyclic_transactions'] = self._detect_cyclic_transactions(graph)
        patterns['fan_in_fan_out'] = self._detect_fan_in_fan_out(graph)
        patterns['mixer_patterns'] = self._detect_mixer_patterns(graph)
        
        return patterns

    def _find_high_frequency_nodes(self, graph, threshold=100):
        high_freq_nodes = []
        for node in graph.nodes():
            degree = graph.degree(node)
            if degree > threshold:
                high_freq_nodes.append((node, degree))
        return sorted(high_freq_nodes, key=lambda x: x[1], reverse=True)

    def _find_high_value_clusters(self, graph, value_threshold=1000):
        high_value_edges = [(u, v, data) for u, v, data in graph.edges(data=True) 
                           if data.get('value', 0) > value_threshold]
        
        subgraph = graph.edge_subgraph([(u, v) for u, v, _ in high_value_edges])
        clusters = list(nx.connected_components(subgraph.to_undirected()))
        
        return [list(cluster) for cluster in clusters if len(cluster) > 1]

    def _detect_cyclic_transactions(self, graph):
        cycles = []
        try:
            simple_cycles = list(nx.simple_cycles(graph))
            for cycle in simple_cycles:
                if len(cycle) <= 5:
                    cycles.append(cycle)
        except:
            pass
        return cycles

    def _detect_fan_in_fan_out(self, graph, ratio_threshold=10):
        suspicious_nodes = []
        for node in graph.nodes():
            in_degree = graph.in_degree(node)
            out_degree = graph.out_degree(node)
            
            if in_degree > 0 and out_degree > 0:
                ratio = max(in_degree, out_degree) / min(in_degree, out_degree)
                if ratio > ratio_threshold:
                    suspicious_nodes.append((node, in_degree, out_degree, ratio))
                    
        return sorted(suspicious_nodes, key=lambda x: x[3], reverse=True)

    def _detect_mixer_patterns(self, graph, min_transactions=10):
        mixer_candidates = []
        for node in graph.nodes():
            in_edges = list(graph.in_edges(node, data=True))
            out_edges = list(graph.out_edges(node, data=True))
            
            if len(in_edges) >= min_transactions and len(out_edges) >= min_transactions:
                in_values = [data['value'] for _, _, data in in_edges]
                out_values = [data['value'] for _, _, data in out_edges]
                
                if abs(sum(in_values) - sum(out_values)) / sum(in_values) < 0.01:
                    mixer_candidates.append(node)
                    
        return mixer_candidates

    def analyze_transaction_network(self, graph):
        analysis = {}
        
        analysis['degree_centrality'] = nx.degree_centrality(graph)
        analysis['betweenness_centrality'] = nx.betweenness_centrality(graph)
        analysis['pagerank'] = nx.pagerank(graph)
        
        try:
            analysis['closeness_centrality'] = nx.closeness_centrality(graph)
        except:
            analysis['closeness_centrality'] = {}
            
        return analysis

    def generate_suspicious_subgraphs(self, graph, top_k=10):
        centrality = nx.betweenness_centrality(graph)
        top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        subgraphs = {}
        for node, _ in top_nodes:
            try:
                ego_graph = nx.ego_graph(graph, node, radius=2)
                subgraphs[node] = ego_graph
            except:
                continue
                
        return subgraphs