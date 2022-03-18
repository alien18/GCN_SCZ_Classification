import os.path as osp
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix


dpath = './data/Main/raw'
C_val = np.logspace(-4, 3, 8)
gamma_val = np.logspace(-3, 2, 6)
coefs_ = []
n_jobs = 5
n_regions = 90

print('Loading data ...')
files = sorted(list(Path(dpath).glob("*.txt")))
features = []

for file in files:
    fc_mat = np.genfromtxt(file)
    fc_vec = np.concatenate([fc_mat[i][:i] for i in range(fc_mat.shape[0])])
    features.append(fc_vec)
features = np.array(features)
labels = np.genfromtxt(osp.join(dpath, 'Labels.csv'), dtype='int32')

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=90)
nested_skf = StratifiedKFold(n_splits=10, shuffle=True)
eval_metrics = np.zeros((skf.n_splits, 3))

for n_fold, (train, test) in enumerate(skf.split(features, labels)):

    print('Processing the No.%i cross-validation in %i-fold CV' % (n_fold + 1, skf.n_splits))
    x_train, y_train = features[train], labels[train]
    x_test, y_test = features[test], labels[test]

    init_clf = SVC(kernel='rbf')
    grid = GridSearchCV(init_clf, {'C': C_val, 'gamma': gamma_val}, cv=nested_skf, scoring='balanced_accuracy',
                        n_jobs=n_jobs)
    grid.fit(x_train, y_train)
    print('    The best parameter C: %.2e and Gamma: %.2e with BAC of %f'
          % (grid.best_params_['C'], grid.best_params_['gamma'], grid.best_score_))
    clf = SVC(kernel='rbf', C=grid.best_params_['C'], gamma=grid.best_params_['gamma'])
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fold_sen = tp / (tp + fn)
    fold_spe = tn / (tn + fp)
    fold_bac = (fold_sen + fold_spe) / 2
    eval_metrics[n_fold, 0] = fold_sen
    eval_metrics[n_fold, 1] = fold_spe
    eval_metrics[n_fold, 2] = fold_bac

eval_df = pd.DataFrame(eval_metrics)
eval_df.columns = ['SEN', 'SPE', 'BAC']
eval_df.index = ['Fold_%02i' % (i + 1) for i in range(skf.n_splits)]
print(eval_df)
print('\nAverage Sensitivity: %.4f±%.4f' % (eval_metrics[:, 0].mean(), eval_metrics[:, 0].std()))
print('Average Specificity: %.4f±%.4f' % (eval_metrics[:, 1].mean(), eval_metrics[:, 1].std()))
print('Average Balanced Accuracy: %.4f±%.4f' % (eval_metrics[:, 2].mean(), eval_metrics[:, 2].std()))