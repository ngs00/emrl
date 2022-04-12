import random
import torch
import torch.nn.functional as F


def get_pairs(batch):
    num_data = len(batch)
    pos_list = list()
    neg_list = list()

    for anc in batch:
        target = anc[1]
        idx = random.sample(range(0, num_data), 2)

        if abs(target - batch[idx[0]][1]) < abs(target - batch[idx[1]][1]):
            pos_list.append(batch[idx[0]])
            neg_list.append(batch[idx[1]])
        else:
            pos_list.append(batch[idx[1]])
            neg_list.append(batch[idx[0]])

    return pos_list, neg_list


def train(emb_net, optimizer, data_loader):
    emb_net.train()
    train_loss = 0

    for anc_x, anc_y, pos_x, pos_y, neg_x, neg_y in data_loader:
        anc_x = anc_x.cuda()
        anc_y = anc_y.cuda()
        pos_x = pos_x.cuda()
        pos_y = pos_y.cuda()
        neg_x = neg_x.cuda()
        neg_y = neg_y.cuda()

        emb_anc = F.normalize(emb_net(anc_x), 2, dim=1)
        emb_pos = F.normalize(emb_net(pos_x), 2, dim=1)
        emb_neg = F.normalize(emb_net(neg_x), 2, dim=1)

        dist_ratio_x = torch.norm(emb_anc - emb_pos, dim=1) / (torch.norm(emb_anc - emb_neg, dim=1) + 1e-5)
        dist_ratio_x = -torch.exp(-dist_ratio_x + 1)
        dist_ratio_y = torch.norm(anc_y - pos_y, dim=1) / (torch.norm(anc_y - neg_y, dim=1) + 1e-5)
        dist_ratio_y = -torch.exp(-dist_ratio_y + 1)

        loss = torch.mean((dist_ratio_x - dist_ratio_y)**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.cpu().item()

    return train_loss / len(data_loader)


def test(emb_net, data_loader):
    emb_net.eval()
    embs = list()

    with torch.no_grad():
        for anc_x, _, _, _, _, _ in data_loader:
            anc_x = anc_x.cuda()

            emb_anc = F.normalize(emb_net(anc_x), 2, dim=1)

            embs.append(emb_anc)

    return torch.cat(embs, dim=0)


def collate(batch):
    list_pos, list_neg = get_pairs(batch)
    list_anc_x = list()
    list_anc_y = list()
    list_pos_x = list()
    list_pos_y = list()
    list_neg_x = list()
    list_neg_y = list()

    for i in range(0, len(batch)):
        list_anc_x.append(batch[i][0].view(1, -1))
        list_anc_y.append(batch[i][1])
        list_pos_x.append(list_pos[i][0].view(1, -1))
        list_pos_y.append(list_pos[i][1])
        list_neg_x.append(list_neg[i][0].view(1, -1))
        list_neg_y.append(list_neg[i][1])

    anc_x = torch.cat(list_anc_x, dim=0)
    anc_y = torch.cat(list_anc_y, dim=0).view(-1, 1)
    pos_x = torch.cat(list_pos_x, dim=0)
    pos_y = torch.cat(list_pos_y, dim=0).view(-1, 1)
    neg_x = torch.cat(list_neg_x, dim=0)
    neg_y = torch.cat(list_neg_y, dim=0).view(-1, 1)

    return anc_x, anc_y, pos_x, pos_y, neg_x, neg_y
