import numpy as np
import pandas as pd
import os.path as osp
import warnings

import torch
import torch.nn.functional as func
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

from Model import GCN
from Dataset import ConnectivityData
from Utils import combat_trans


def GCN_train(loader):
    model.train()

    train_loss_all = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        train_loss = func.cross_entropy(output, data.y)
        train_loss.backward()
        train_loss_all += data.num_graphs * train_loss.item()
        optimizer.step()
    return train_loss_all / len(train_dataset)


def GCN_test(loader):
    model.eval()

    pred = []
    label = []
    for data in loader:
        data = data.to(device)
        output = model(data)
        pred.append(func.softmax(output, dim=1).max(dim=1)[1])
        label.append(data.y)

    y_pred = torch.cat(pred, dim=0).cpu().detach().numpy()
    y_true = torch.cat(label, dim=0).cpu().detach().numpy()
    tn, fp, fn, tp = confusion_matrix(y_pred, y_true).ravel()
    epoch_sen = tp / (tp + fn)
    epoch_spe = tn / (tn + fp)
    epoch_bac = (epoch_sen + epoch_spe) / 2
    return epoch_sen, epoch_spe, epoch_bac


warnings.filterwarnings("ignore")
dataset = ConnectivityData('./data_demo/Main')

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=99)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
labels = np.genfromtxt(osp.join(dataset.raw_dir, 'Labels.csv'))
covars = pd.read_csv(osp.join(dataset.raw_dir, 'covars.csv'), delimiter=',')

eval_metrics = np.zeros((skf.n_splits, 3))

for n_fold, (train, test) in enumerate(skf.split(labels, labels)):

    model = GCN(dataset.num_features, dataset.num_classes, 6).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    train_dataset, test_dataset = dataset[train.tolist()], dataset[test.tolist()]

    case, batch = covars[['case']], covars[['batch']]
    case_train, case_test = case.iloc[train], case.iloc[test]
    batch_train, batch_test = batch.iloc[train], batch.iloc[test]
    train_dataset_harmonized, test_dataset_harmonized = combat_trans(train_dataset,
                                                                     test_dataset,
                                                                     batch_train,
                                                                     batch_test,
                                                                     case_train,
                                                                     case_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    for epoch in range(2):
        t_loss = GCN_train(train_loader)
        test_sen, test_spe, test_bac = GCN_test(test_loader)

        print('CV: {:03d}, Epoch: {:03d}, Train Loss: {:.5f}, Test BAC: {:.5f}, TEST SEN: {:.5f}, TEST SPE: {:.5f}'.
              format(n_fold + 1, epoch + 1, t_loss, test_bac, test_sen, test_spe))

    eval_metrics[n_fold, 0] = test_sen
    eval_metrics[n_fold, 1] = test_spe
    eval_metrics[n_fold, 2] = test_bac

eval_df = pd.DataFrame(eval_metrics)
eval_df.columns = ['SEN', 'SPE', 'BAC']
eval_df.index = ['Fold_%02i' % (i + 1) for i in range(skf.n_splits)]
print(eval_df)
print('Average Sensitivity: %.4f±%.4f' % (eval_metrics[:, 0].mean(), eval_metrics[:, 0].std()))
print('Average Specificity: %.4f±%.4f' % (eval_metrics[:, 1].mean(), eval_metrics[:, 1].std()))
print('Average Balanced Accuracy: %.4f±%.4f' % (eval_metrics[:, 2].mean(), eval_metrics[:, 2].std()))
