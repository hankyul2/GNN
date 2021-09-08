"""
This is tutorial from https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html
If you find it useful go to above link, which has full description and code.

This files include EdgeConv Class definition. EdgeConv Class consist of 3 steps
EdgeConv: Update Node features by Max(f([x_u, x_v - x_u])) V is U's k-neighbors

1. cat feature
2. linear transform
3. Aggregate

function call procedures: forward -> propagate -> message -> aggregate -> update
"""
import torch
from torch import nn
from torch_geometric.nn import MessagePassing, knn_graph


class EdgeConv(MessagePassing):
    def __init__(self, in_channels=50, out_channels=10):
        super(EdgeConv, self).__init__(aggr='max')
        self.mlp = nn.Sequential(nn.Linear(in_channels * 2, out_channels), nn.ReLU(), nn.Linear(out_channels, out_channels))

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        return self.mlp(torch.cat([x_i, x_j-x_i], dim=1))


class DynamicEdgeConv(EdgeConv):
    def __init__(self, in_channels, out_channels, k=6):
        super(DynamicEdgeConv, self).__init__(in_channels, out_channels)
        self.k = k

    def forward(self, x, batch=None):
        edge_index = knn_graph(x, k=self.k, batch=batch, flow=self.flow, loop=False)
        return super().forward(x, edge_index)


if __name__ == '__main__':
    x = torch.rand(10, 10)
    conv = DynamicEdgeConv(10, 10)
    out = conv(x)

    print('x shape {}'.format(x.shape))
    print('out shape {}'.format(out.shape))