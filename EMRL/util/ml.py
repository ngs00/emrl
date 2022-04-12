import numpy
import torch
import xgboost as xgb
import nn.util as nutil
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from nn.models import FNN


def split_train_val(data, ratio):
    n_train = int(ratio * data.shape[0])

    numpy.random.shuffle(data)

    return data[:n_train, :], data[n_train:, :]


def eval_lr(train_x, train_y, test_x, test_y):
    model = LinearRegression().fit(train_x, train_y)
    preds = model.predict(test_x).reshape(-1, 1)
    mae = numpy.mean(numpy.abs(test_y - preds))
    r2 = r2_score(test_y, preds)

    return model, mae, r2, preds


def eval_xgb(train_x, train_y, test_x, test_y):
    train_data, val_data = split_train_val(numpy.hstack([train_x, train_y]), ratio=0.8)
    train_x = train_data[:, :-1]
    train_y = train_data[:, -1].reshape(-1, 1)
    val_x = val_data[:, :-1]
    val_y = val_data[:, -1].reshape(-1, 1)
    min_val_error = 1e+8

    for d in range(3, 8):
        for n in [100, 150, 200, 300, 400]:
            model = xgb.XGBRegressor(max_depth=d, n_estimators=n, subsample=0.8)
            model.fit(train_x, train_y, eval_metric='mae', eval_set=[(val_x, val_y)])
            preds = model.predict(val_x).reshape(-1, 1)
            val_error = numpy.mean(numpy.abs(val_y - preds))
            print('d={}\tn={}\tMAE: {:.4f}'.format(d, n, val_error))

            if val_error < min_val_error:
                min_val_error = val_error
                opt_d = d
                opt_n = n

    model = xgb.XGBRegressor(max_depth=opt_d, n_estimators=opt_n, subsample=0.8)
    model.fit(train_x, train_y, eval_metric='mae')
    preds = model.predict(test_x).reshape(-1, 1)
    print(test_y.shape, preds.shape)
    mae = numpy.mean(numpy.abs(test_y - preds))
    r2 = r2_score(test_y, preds)

    return model, mae, r2, preds


def eval_fnn(train_x, train_y, test_x, test_y, dim_emb):
    train_data, val_data = split_train_val(numpy.hstack([train_x, train_y]), ratio=0.8)
    train_x = torch.tensor(train_data[:, :-1], dtype=torch.float)
    train_y = torch.tensor(train_data[:, -1], dtype=torch.float).view(-1, 1)
    val_x = torch.tensor(val_data[:, :-1], dtype=torch.float)
    val_y = torch.tensor(val_data[:, -1], dtype=torch.float).view(-1, 1)
    test_x = torch.tensor(test_x, dtype=torch.float)
    test_y = torch.tensor(test_y, dtype=torch.float).view(-1, 1)

    batch_size = 32
    n_epochs = 500
    min_val_loss = 1e+8
    opt_model_params = None

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size)

    model = FNN(dim_emb, 1).cuda()
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

    for epoch in range(0, n_epochs):
        train_loss = nutil.train(model, train_loader, optimizer, criterion)
        val_loss, _ = nutil.test(model, val_loader, criterion)
        print('Epoch [{}/{}]\tTrain loss: {:.4f}\tVal loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss, val_loss))

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            opt_model_params = model.state_dict()

    model.load_state_dict(opt_model_params)
    mae, preds = nutil.test(model, test_loader, criterion)

    return model, mae, r2_score(test_y, preds), preds
