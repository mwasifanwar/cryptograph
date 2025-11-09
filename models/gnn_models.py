# models/gnn_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_mean_pool
import torch_geometric

class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        
        if batch is not None:
            x = global_mean_pool(x, batch)
            
        x = self.classifier(x)
        return x

class GATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=8, dropout=0.3):
        super(GATModel, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.gat3 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None):
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat3(x, edge_index))
        
        if batch is not None:
            x = global_mean_pool(x, batch)
            
        x = self.classifier(x)
        return x

class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(GraphSAGE, self).__init__()
        self.sage1 = SAGEConv(input_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)
        self.sage3 = SAGEConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None):
        x = F.relu(self.sage1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.sage2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.sage3(x, edge_index))
        
        if batch is not None:
            x = global_mean_pool(x, batch)
            
        x = self.classifier(x)
        return x

class MultiModalBlockchainModel(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, output_dim):
        super(MultiModalBlockchainModel, self).__init__()
        
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.gnn_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim, heads=4, concat=False),
            GATConv(hidden_dim, hidden_dim, heads=4, concat=False),
            GATConv(hidden_dim, hidden_dim, heads=1, concat=False)
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, edge_index, edge_attr, batch=None):
        x = self.node_encoder(x)
        edge_emb = self.edge_encoder(edge_attr)
        
        for conv in self.gnn_layers:
            x = F.relu(conv(x, edge_index, edge_emb))
            x = F.dropout(x, p=0.3, training=self.training)
        
        if batch is not None:
            x = global_mean_pool(x, batch)
            
        graph_embedding = x
        
        x = self.classifier(graph_embedding)
        return x, graph_embedding