"""
    GAT (Graph Attention Network) model definition
    for graph classification on MUTAG dataset
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv as PyGGATConv
from torch_geometric.nn import global_mean_pool, global_max_pool

import curves

__all__ = ['GAT']


class GATBase(nn.Module):
    """
    Base GAT model for graph classification.

    Architecture:
    - Node embedding layer
    - 3 GAT layers with multi-head attention
    - Global pooling (mean + max)
    - MLP classifier
    """

    def __init__(self, num_classes, num_node_features=7, hidden_dim=64, heads=4):
        super(GATBase, self).__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.heads = heads

        # Node feature embedding
        self.node_emb = nn.Linear(num_node_features, hidden_dim)

        # GAT layers with multi-head attention
        # Layer 1: 64 -> 64*4 (concatenate 4 heads)
        self.gat1 = PyGGATConv(hidden_dim, hidden_dim, heads=heads,
                               concat=True, dropout=0.3)

        # Layer 2: 256 -> 64*4
        self.gat2 = PyGGATConv(hidden_dim * heads, hidden_dim, heads=heads,
                               concat=True, dropout=0.3)

        # Layer 3: 256 -> 64 (single head, no concatenation)
        self.gat3 = PyGGATConv(hidden_dim * heads, hidden_dim, heads=1,
                               concat=False, dropout=0.3)

        # Global pooling combines mean and max -> 128 features
        # Classifier MLP
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # 128 -> 64
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)  # 64 -> 2
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, batch):
        """
        Forward pass for graph classification.

        Args:
            x: Node features [num_nodes, num_node_features]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment vector [num_nodes]

        Returns:
            Logits [batch_size, num_classes]
        """
        # Node embedding
        x = self.node_emb(x)

        # GAT layers with ELU activation
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        x = F.elu(self.gat3(x, edge_index))

        # Global pooling: combine mean and max
        x_mean = global_mean_pool(x, batch)  # [batch_size, hidden_dim]
        x_max = global_max_pool(x, batch)    # [batch_size, hidden_dim]
        x = torch.cat([x_mean, x_max], dim=1)  # [batch_size, hidden_dim*2]

        # Classification
        x = self.classifier(x)

        return x


class GATCurve(nn.Module):
    """
    GAT curve model for mode connectivity.

    Implements the same architecture as GATBase, but with all linear/GAT layers
    replaced by curve versions that interpolate parameters along a Bezier curve.
    """

    def __init__(self, num_classes, fix_points, num_node_features=7,
                 hidden_dim=64, heads=4):
        super(GATCurve, self).__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.heads = heads

        # Node feature embedding (curve version)
        self.node_emb = curves.Linear(num_node_features, hidden_dim, fix_points=fix_points)

        # GAT layers with curve parameters
        self.gat1 = curves.GATConv(hidden_dim, hidden_dim, fix_points=fix_points,
                                    heads=heads, concat=True, dropout=0.3)
        self.gat2 = curves.GATConv(hidden_dim * heads, hidden_dim, fix_points=fix_points,
                                    heads=heads, concat=True, dropout=0.3)
        self.gat3 = curves.GATConv(hidden_dim * heads, hidden_dim, fix_points=fix_points,
                                    heads=1, concat=False, dropout=0.3)

        # Classifier (curve version)
        self.fc1 = curves.Linear(hidden_dim * 2, hidden_dim, fix_points=fix_points)
        self.fc2 = curves.Linear(hidden_dim, num_classes, fix_points=fix_points)

        # Dropout layers (not parameterized, same across curve)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x, edge_index, batch, coeffs_t):
        """
        Forward pass for curve model.

        Args:
            x: Node features [num_nodes, num_node_features]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment vector [num_nodes]
            coeffs_t: Curve coefficients at point t

        Returns:
            Logits [batch_size, num_classes]
        """
        # Node embedding with curve parameters
        x = self.node_emb(x, coeffs_t)

        # GAT layers with ELU activation and dropout
        x = F.elu(self.gat1(x, edge_index, coeffs_t))
        x = self.dropout1(x)

        x = F.elu(self.gat2(x, edge_index, coeffs_t))
        x = self.dropout1(x)

        x = F.elu(self.gat3(x, edge_index, coeffs_t))

        # Global pooling: combine mean and max
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        # Classification with curve parameters
        x = F.relu(self.fc1(x, coeffs_t))
        x = self.dropout2(x)
        x = self.fc2(x, coeffs_t)

        return x


class GAT:
    """
    GAT model wrapper for compatibility with training scripts.
    """
    base = GATBase
    curve = GATCurve
    kwargs = {
        'num_node_features': 7,  # MUTAG has 7 discrete node labels
        'hidden_dim': 64,
        'heads': 4
    }
