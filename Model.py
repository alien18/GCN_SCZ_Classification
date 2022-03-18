import torch
import torch.nn.functional as func
from torch_geometric.nn import ChebConv, global_mean_pool


class GCN(torch.nn.Module):
    """GCN model(network architecture can be modified)"""

    def __init__(self,
                 num_features,
                 num_classes,
                 k_order,
                 dropout=.5):
        super(GCN, self).__init__()

        self.p = dropout

        self.conv1 = ChebConv(int(num_features), 64, K=k_order)
        self.conv2 = ChebConv(64, 64, K=k_order)
        self.conv3 = ChebConv(64, 128, K=k_order)

        self.lin1 = torch.nn.Linear(128, int(num_classes))

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch

        x = func.relu(self.conv1(x, edge_index, edge_attr))
        x = func.dropout(x, p=self.p, training=self.training)
        x = func.relu(self.conv2(x, edge_index, edge_attr))
        x = func.dropout(x, p=self.p, training=self.training)
        x = func.relu(self.conv3(x, edge_index, edge_attr))

        x = global_mean_pool(x, batch)
        x = self.lin1(x)
        return x
