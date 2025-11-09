# utils/visualization_utils.py
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import numpy as np

class VisualizationUtils:
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def plot_transaction_network(self, graph, highlight_nodes=None, title="Transaction Network"):
        pos = nx.spring_layout(graph, k=1, iterations=50)
        
        plt.figure(figsize=(12, 10))
        
        node_colors = []
        for node in graph.nodes():
            if highlight_nodes and node in highlight_nodes:
                node_colors.append('red')
            else:
                node_colors.append('lightblue')
                
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, 
                              node_size=50, alpha=0.7)
        nx.draw_networkx_edges(graph, pos, alpha=0.2, arrows=True, 
                              arrowsize=10, edge_color='gray')
        
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        return plt.gcf()
    
    def plot_risk_distribution(self, risk_scores, title="Risk Score Distribution"):
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=risk_scores,
            nbinsx=50,
            name='Risk Scores',
            opacity=0.7,
            marker_color='red'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Risk Score',
            yaxis_title='Frequency',
            template='plotly_white'
        )
        
        return fig
    
    def create_network_dashboard(self, graph, risk_scores, top_suspicious):
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "scatter"}, {"type": "histogram"}],
                   [{"type": "bar"}, {"type": "pie"}]],
            subplot_titles=('Network Overview', 'Risk Distribution', 
                          'Top Suspicious Nodes', 'Risk Level Breakdown')
        )
        
        return fig
    
    def plot_temporal_patterns(self, transactions_df, title="Temporal Transaction Patterns"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        hourly_volume = transactions_df.groupby(transactions_df['timestamp'].dt.hour)['value_eth'].sum()
        ax1.plot(hourly_volume.index, hourly_volume.values, marker='o')
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Transaction Volume (ETH)')
        ax1.set_title('Hourly Transaction Volume')
        
        daily_pattern = transactions_df.groupby(transactions_df['timestamp'].dt.dayofweek)['value_eth'].sum()
        ax2.bar(daily_pattern.index, daily_pattern.values)
        ax2.set_xlabel('Day of Week')
        ax2.set_ylabel('Transaction Volume (ETH)')
        ax2.set_title('Weekly Transaction Pattern')
        
        plt.tight_layout()
        return fig