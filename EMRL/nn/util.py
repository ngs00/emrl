import numpy
import torch


def train(model, data_loader, optimizer, criterion):
    sum_train_losses = 0

    for data, targets in data_loader:
        data = data.cuda()
        targets = targets.cuda()

        preds = model(data)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_train_losses += loss.item()

    return sum_train_losses / len(data_loader)


def test(model, data_loader, criterion):
    model.eval()
    list_preds = list()
    test_loss = 0

    with torch.no_grad():
        for data, targets in data_loader:
            data = data.cuda()
            targets = targets.view(-1, 1).cuda()

            preds = model(data)
            loss = criterion(preds, targets)
            test_loss += loss.item()

            list_preds.append(preds.cpu().detach().numpy())

    return test_loss / len(data_loader), numpy.vstack(list_preds)
