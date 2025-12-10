"""
    GIN (Graph Isomorphism Network) model definition
    for graph classification on MUTAG dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv as PyGGINConv
from torch_geometric.nn import global_mean_pool, global_max_pool

import curves

__all__ = ['GIN']


class GINBase(nn.Module):
    """
    Base GIN model for graph classification.

    Architecture:
    - Node embedding layer
    - 3 GIN layers with 2-layer MLP
    - Global pooling (mean + max)
    - MLP classifier
    """

    def __init__(self, num_classes, num_node_features=7, hidden_dim=64):
        super(GINBase, self).__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # Node feature embedding
        self.node_emb = nn.Linear(num_node_features, hidden_dim)

        # GIN layers with 2-layer MLP
        # Each MLP: hidden_dim -> 2*hidden_dim -> hidden_dim
        nn_1 = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2 * hidden_dim, hidden_dim)
        )
        self.gin1 = PyGGINConv(nn_1, train_eps=True)

        nn_2 = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2 * hidden_dim, hidden_dim)
        )
        self.gin2 = PyGGINConv(nn_2, train_eps=True)

        nn_3 = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2 * hidden_dim, hidden_dim)
        )
        self.gin3 = PyGGINConv(nn_3, train_eps=True)

        # Dropout
        self.dropout = nn.Dropout(0.3)

        # Global pooling combines mean and max -> 128 features
        # Classifier MLP
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # 128 -> 64
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)  # 64 -> num_classes
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

        # GIN layers with ELU activation and dropout
        x = F.elu(self.dropout(self.gin1(x, edge_index)))
        x = F.elu(self.dropout(self.gin2(x, edge_index)))
        x = F.elu(self.dropout(self.gin3(x, edge_index)))

        # Global pooling: combine mean and max
        x_mean = global_mean_pool(x, batch)  # [batch_size, hidden_dim]
        x_max = global_max_pool(x, batch)    # [batch_size, hidden_dim]
        x = torch.cat([x_mean, x_max], dim=1)  # [batch_size, hidden_dim*2]

        # Classification
        x = self.classifier(x)

        return x


class GINCurve(nn.Module):
    """
    GIN curve model for mode connectivity.

    Implements the same architecture as GINBase, but with all linear/GIN layers
    replaced by curve versions that interpolate parameters along a Bezier curve.
    """

    def __init__(self, num_classes, fix_points, num_node_features=7, hidden_dim=64):
        super(GINCurve, self).__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # Node feature embedding (curve version)
        self.node_emb = curves.Linear(num_node_features, hidden_dim, fix_points=fix_points)

        # GIN layers with curve parameters
        # Each GIN layer has a 2-layer MLP: hidden_dim -> 2*hidden_dim -> hidden_dim
        self.gin1 = curves.GINConv(hidden_dim, hidden_dim, 2 * hidden_dim,
                                    fix_points=fix_points, train_eps=True)
        self.gin2 = curves.GINConv(hidden_dim, hidden_dim, 2 * hidden_dim,
                                    fix_points=fix_points, train_eps=True)
        self.gin3 = curves.GINConv(hidden_dim, hidden_dim, 2 * hidden_dim,
                                    fix_points=fix_points, train_eps=True)

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

        # GIN layers with ELU activation and dropout
        x = F.elu(self.dropout1(self.gin1(x, edge_index, coeffs_t)))
        x = F.elu(self.dropout1(self.gin2(x, edge_index, coeffs_t)))
        x = F.elu(self.dropout1(self.gin3(x, edge_index, coeffs_t)))

        # Global pooling: combine mean and max
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        # Classification with curve parameters
        x = F.relu(self.fc1(x, coeffs_t))
        x = self.dropout2(x)
        x = self.fc2(x, coeffs_t)

        return x


class GIN:
    """
    GIN model wrapper for compatibility with training scripts.
    """
    base = GINBase
    curve = GINCurve
    kwargs = {
        'num_node_features': 7,  # MUTAG has 7 discrete node labels
        'hidden_dim': 64
    }
