"""
This is tutorial from https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html
If you find it useful go to above link, which has full description and code.

This files include GCNConv Class definition. GCNConv Class consist of 5 steps

1. add self-loop index
2. linear transformation
3. calculate normalization coefficient
4. Normalize features
5. Aggregate
"""

import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class GCNConv(MessagePassing):
    def __init__(self, in_channels: int = 10, out_channels: int = 10):
        super(GCNConv, self).__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.tensor, edge_index: torch.tensor):
        # step 1. add self-loop
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # step 2. linear transform
        x = self.lin(x)

        # step 3. calculate norm coef
        row, col = edge_index
        deg = degree(col, x.size(0), x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # step 4. normalize
        return norm.view(-1, 1) * x_j


if __name__ == '__main__':
    x = torch.rand(10, 10)
    edge_index = torch.randint(0, 10, size=(2, 10))
    conv = GCNConv(10, 10)
    out = conv(x, edge_index)

    print('x shape {}'.format(x.shape))
    print('edge_index shape {}'.format(edge_index.shape))
    print('out shape {}'.format(out.shape))
