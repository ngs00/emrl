import numpy
import random
import torch
import chem.base as cb
import nn.dml_tgnn as dml
import util.ml as ml
from torch.utils.data import DataLoader
from chem.crystal import load_dataset
from nn.models import TGNN


# Experiment settings
dataset_name = 'hoip'
target_idx = 1
n_epochs = 1000
batch_size = 32
dim_emb = 128
n_repeats = 10

# Prediction results
list_mae_lr = list()
list_mae_fnn = list()
list_mae_xgb = list()
list_r2_lr = list()
list_r2_fnn = list()
list_r2_xgb = list()

# Load dataset
dataset = load_dataset('datasets/' + dataset_name, 'datasets/' + dataset_name + '/metadata.xlsx',
                       target_idx=target_idx, radius=5)

# Evaluation
for i in range(0, n_repeats):
    # Split training and test datasets
    random.shuffle(dataset)
    train_data = dataset[:int(0.8 * len(dataset))]
    test_data = dataset[int(0.8 * len(dataset)):]
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=dml.collate)
    test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=dml.collate)

    # Configuration embedding network and optimizer
    emb_net = TGNN(cb.n_elem_feats, cb.n_bond_feats, dim_emb).cuda()
    optimizer = torch.optim.Adam(emb_net.parameters(), lr=5e-4, weight_decay=5e-6)

    # Training of embedding network
    for epoch in range(0, n_epochs):
        train_loss = dml.train(emb_net, optimizer, train_loader)
        print('Epoch [{}/{}]\tTrain loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss))

    # Calculate embeddings of the training dataset
    train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=dml.collate)
    emb_train = dml.test(emb_net, train_loader).cpu().numpy()
    train_y = numpy.array([x.y.item() for x in train_data]).reshape((-1, 1))
    train_ids = numpy.array([x.idx for x in train_data]).reshape((-1, 1))
    train_embs = numpy.hstack([emb_train, train_y, train_ids])

    # Calculate embeddings of the test dataset
    emb_test = dml.test(emb_net, test_loader).cpu().numpy()
    test_y = numpy.array([x.y.item() for x in test_data]).reshape((-1, 1))
    test_ids = numpy.array([x.idx for x in test_data]).reshape((-1, 1))
    test_embs = numpy.hstack([emb_test, test_y, test_ids])

    # Save embedding results
    numpy.savetxt('res/emb/emb_' + dataset_name + str(i) + '_train.csv', train_embs, delimiter=',')
    numpy.savetxt('res/emb/emb_' + dataset_name + str(i) + '_test.csv', test_embs, delimiter=',')
    torch.save(emb_net.state_dict(), 'res/trained_model/emb_net_' + dataset_name + str(i) + '.pt')

    # Predict target property using the generated embeddings
    model_lr, mae_lr, r2_lr, preds_lr = ml.eval_lr(emb_train, train_y, emb_test, test_y)
    model_xgb, mae_xgb, r2_xgb, preds_xgb = ml.eval_xgb(emb_train, train_y, emb_test, test_y)
    model_fnn, mae_fnn, r2_fnn, preds_fnn = ml.eval_fnn(emb_train, train_y, emb_test, test_y, dim_emb)
    numpy.savetxt('res/pred/pred_' + dataset_name + str(i) + '_lr.csv', numpy.hstack([test_ids, test_y, preds_lr]),
                  delimiter=',')
    numpy.savetxt('res/pred/pred_' + dataset_name + str(i) + '_xgb.csv', numpy.hstack([test_ids, test_y, preds_xgb]),
                  delimiter=',')
    numpy.savetxt('res/pred/pred_' + dataset_name + str(i) + '_fnn.csv', numpy.hstack([test_ids, test_y, preds_fnn]),
                  delimiter=',')

    # Save evaluations metrics
    list_mae_lr.append(mae_lr)
    list_mae_fnn.append(mae_fnn)
    list_mae_xgb.append(mae_xgb)
    list_r2_lr.append(r2_lr)
    list_r2_fnn.append(r2_fnn)
    list_r2_xgb.append(r2_xgb)

    print('------------------------')
    print(mae_lr, r2_lr)
    print(mae_xgb, r2_xgb)
    print(mae_fnn, r2_fnn)

# Show evaluations results
print('------------------------')
print(numpy.mean(list_mae_lr), numpy.std(list_mae_lr), numpy.mean(list_r2_lr), numpy.std(list_r2_lr))
print(numpy.mean(list_mae_fnn), numpy.std(list_mae_fnn), numpy.mean(list_r2_fnn), numpy.std(list_r2_fnn))
print(numpy.mean(list_mae_xgb), numpy.std(list_mae_xgb), numpy.mean(list_r2_xgb), numpy.std(list_r2_xgb))
