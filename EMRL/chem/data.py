import torch


class TuplewiseCrystal:
    def __init__(self, pair, y, idx):
        self.pair = pair
        self.y = y
        self.idx = idx


class Batch:
    def __init__(self, x_pair, idx_pair, y):
        self.x_pair = x_pair
        self.idx_pair = idx_pair
        self.y = y

    def cuda(self):
        self.x_pair = self.x_pair.cuda()
        self.idx_pair = self.idx_pair.cuda()
        self.y = self.y.cuda()

    def free(self):
        del self.x_pair
        del self.idx_pair
        del self.y

    @staticmethod
    def from_data_list(data_list):
        list_pairs = list()
        list_idx_pairs = list()
        list_targets = list()

        for data in data_list:
            list_pairs.append(data.pair)
            list_idx_pairs.append(data.pair.shape[0])
            list_targets.append(data.y)

        pairs = torch.cat(list_pairs, dim=0)
        idx_pairs = torch.tensor(list_idx_pairs, dtype=torch.long).view(-1, 1)
        targets = torch.cat(list_targets, dim=0)

        return Batch(pairs, idx_pairs, targets)
