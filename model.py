# model.py
import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_add_pool

class GilgameshPredictorV2(torch.nn.Module):
    def __init__(self, num_node_features, protein_emb_dim=480, hidden_dim=128, dropout=0.4):
        super().__init__()

        self.conv1 = GINConv(
            nn.Sequential(nn.Linear(num_node_features, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                          nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        
        self.conv2 = GINConv(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                          nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        
        self.conv3 = GINConv(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                          nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))

        self.protein_projector = nn.Sequential(
            nn.Linear(protein_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.predict = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GIN Convolution
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        
        # Pooling
        x_graph = global_add_pool(x, batch)

        # Protein
        p_emb = data.protein_emb.squeeze(1)
        x_prot = self.protein_projector(p_emb)

        # Fusion
        x_combined = torch.cat([x_graph, x_prot], dim=1)
        
        return self.predict(x_combined).view(-1)