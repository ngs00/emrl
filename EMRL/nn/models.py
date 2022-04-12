import torch
import torch.nn as nn
import torch.nn.functional as F


class FNN(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(dim_in, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, dim_out)

    def forward(self, x):
        h = F.relu(self.bn1(self.fc1(x)))
        h = F.relu(self.bn2(self.fc2(h)))
        h = self.fc3(h)

        return h


class PairNet(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(PairNet, self).__init__()
        self.fc1 = nn.Linear(dim_in, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, dim_out)

    def forward(self, x):
        h = F.relu(self.bn1(self.fc1(x)))
        out = self.fc3(h)

        return out


class TGNN(nn.Module):
    def __init__(self, num_atom_feats, num_bond_feats, dim_target):
        super(TGNN, self).__init__()
        self.num_atom_feats = num_atom_feats
        self.num_bond_feats = num_bond_feats
        self.atom_net = nn.Linear(self.num_atom_feats, 128)
        self.bond_net = nn.Linear(self.num_bond_feats, 64)
        self.pair_net = PairNet(2 * 128 + 64, 128)
        self.attn = nn.Linear(128, 1)

        self.fc1 = nn.Linear(128, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, dim_target)

    def forward(self, x_pair, idx_pairs):
        atom_emb1 = F.relu(self.atom_net(x_pair[:, :self.num_atom_feats]))
        atom_emb2 = F.relu(self.atom_net(x_pair[:, self.num_atom_feats:2*self.num_atom_feats]))
        bond_emb = F.relu(self.bond_net(x_pair[:, 2*self.num_atom_feats:]))
        h_pair = self.readout(self.pair_net(torch.cat([atom_emb1, atom_emb2, bond_emb], dim=1)), idx_pairs)
        h = F.relu(self.bn1(self.fc1(h_pair)))
        out = self.fc2(h)

        return out

    def readout(self, x, idx):
        h = torch.empty((idx.shape[0], x.shape[1]), dtype=torch.float).cuda()
        pos = 0

        for i in range(0, idx.shape[0]):
            h[i, :] = torch.mean(x[pos:pos+idx[i], :], dim=0)
            pos += idx[i]

        return h
