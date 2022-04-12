import random
import torch
from chem.data import Batch
import torch.nn.functional as F


def get_pairs(batch):
    num_data = len(batch)
    pos_list = list()
    neg_list = list()

    for anc in batch:
        target = anc.y
        idx = random.sample(range(0, num_data), 2)

        if abs(target - batch[idx[0]].y) < abs(target - batch[idx[1]].y):
            pos_list.append(batch[idx[0]])
            neg_list.append(batch[idx[1]])
        else:
            pos_list.append(batch[idx[1]])
            neg_list.append(batch[idx[0]])

    return pos_list, neg_list


def train(emb_net, optimizer, data_loader):
    emb_net.train()
    train_loss = 0

    for i, (anc, pos, neg) in enumerate(data_loader):
        anc.cuda()
        pos.cuda()
        neg.cuda()

        emb_anc = F.normalize(emb_net(anc.x_pair, anc.idx_pair), 2, dim=1)
        emb_pos = F.normalize(emb_net(pos.x_pair, pos.idx_pair), 2, dim=1)
        emb_neg = F.normalize(emb_net(neg.x_pair, neg.idx_pair), 2, dim=1)

        dist_ratio_x = torch.norm(emb_anc - emb_pos, dim=1) / (torch.norm(emb_anc - emb_neg, dim=1) + 1e-5)
        dist_ratio_x = -torch.exp(-dist_ratio_x + 1)
        dist_ratio_y = torch.norm(anc.y - pos.y, dim=1) / (torch.norm(anc.y - neg.y, dim=1) + 1e-5)
        dist_ratio_y = -torch.exp(-dist_ratio_y + 1)

        loss = torch.mean(torch.clamp((dist_ratio_x - dist_ratio_y)**2 - 0.2, min=0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.cpu().item()

        if (i + 1) % 20 == 0:
            print('[' + str(i + 1) + '/' + str(len(data_loader)) + ']')

    return train_loss / len(data_loader)


def test(emb_net, data_loader):
    emb_net.eval()
    embs = list()

    with torch.no_grad():
        for anc, _, _ in data_loader:
            anc.cuda()

            emb = F.normalize(emb_net(anc.x_pair, anc.idx_pair), 2, dim=1)
            embs.append(emb)

    return torch.cat(embs, dim=0)


def collate(batch):
    pos_list, neg_list = get_pairs(batch)

    return Batch.from_data_list(batch), Batch.from_data_list(pos_list), Batch.from_data_list(neg_list)
