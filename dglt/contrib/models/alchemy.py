import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.nn import NNConv, Set2Set

class Alchemy(torch.nn.Module):
    def __init__(self,
                 node_input_dim=133,
                 edge_input_dim=169,
                 output_dim=1,
                 node_hidden_dim=64,
                 edge_hidden_dim=128,
                 num_step_message_passing=6,
                 num_step_set2set=1):
        super(Alchemy, self).__init__()
        self.num_step_message_passing = num_step_message_passing
        self.lin0 = nn.Linear(node_input_dim, node_hidden_dim)
        edge_network = nn.Sequential(
                nn.Linear(edge_input_dim, edge_hidden_dim),
                nn.ReLU(),
                nn.Linear(edge_hidden_dim, node_hidden_dim * node_hidden_dim)
                )
        self.conv = NNConv(node_hidden_dim, node_hidden_dim, edge_network, aggr='mean', root_weight=False)
        self.gru = nn.GRU(node_hidden_dim, node_hidden_dim)

        self.set2set = Set2Set(node_hidden_dim, processing_steps=num_step_set2set)
        self.lin1 = nn.Linear(2 * node_hidden_dim, node_hidden_dim)
        self.lin2 = nn.Linear(node_hidden_dim, output_dim)
        self.ffn = nn.Sequential(self.lin1, self.lin2)

    def forward(self, data, features_batch):
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a = data
        print(f_atoms.shape)
        print(f_bonds.shape)
        batch = []
        for idx, item in enumerate(a_scope):
            batch.extend([idx for _ in range(item[1])])
        batch = [0] + batch  # fix padding
        batch = torch.LongTensor(batch)
        edge_index = torch.stack([b2a, b2a[b2revb]])
        edge_attr = f_bonds
        if next(self.parameters()).is_cuda:
            f_atoms, edge_attr, edge_index = f_atoms.cuda(), edge_attr.cuda(), edge_index.cuda()
            batch = batch.cuda()
        print(batch.shape)
        out = F.relu(self.lin0(f_atoms))
        h = out.unsqueeze(0)

        for i in range(self.num_step_message_passing):
            m = F.relu(self.conv(out, edge_index, edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, batch)
        # out = torch.cat(out, features_batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out

if __name__ == '__main__':
    mpnn = Alchemy()
    print(mpnn)
