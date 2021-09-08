import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from torch_geometric.datasets import Planetoid


class Net(torch.nn.Module):
    def __init__(self, in_features, nclass):
        super().__init__()
        self.conv1 = GCNConv(in_features, 16)
        self.conv2 = GCNConv(16, nclass)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)

def run(args):
    device = torch.device('cpu')

    # step 1. load dataset
    dataset = Planetoid(root='Data', name='Cora')
    data = dataset[0].to(device)

    # step 2. load model
    model = Net(dataset.num_node_features, dataset.num_classes).to(device)

    # step 3. prepare training tool
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    # step 4. train
    print_freq = 20
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch!=0 and epoch % print_freq == 0:
            print('epoch: {}/{} loss: {:6.4f}'.format(epoch+1, 200, loss.detach().item()))

    # step 5. evaluate
    model.eval()
    _, pred = model(data).max(dim=1)
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / int(data.test_mask.sum())
    print('Accuracy: {:.4f}'.format(acc))
