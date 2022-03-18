import os.path as osp
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix


dpath = './data/Main/raw'
C_val = np.logspace(-4, 3, 8)
n_jobs = 5

print('Loading data ...')
files = sorted(list(Path(dpath).glob("*.txt")))
features = []

for file in files:
    fc_mat = np.genfromtxt(file)
    fc_vec = np.concatenate([fc_mat[i][:i] for i in range(fc_mat.shape[0])])
    features.append(fc_vec)
features = np.array(features)
labels = np.genfromtxt(osp.join(dpath, 'Labels.csv'), dtype='int32')
sites = np.genfromtxt(osp.join(dpath, 'sites.csv'))

logo = LeaveOneGroupOut()
nested_skf = StratifiedKFold(n_splits=10, shuffle=True)
n_sites = np.unique(sites).shape[0]
eval_metrics = np.zeros((n_sites - 1, 3)) # not testing on site 6 (HC site (Dataset 2 in manuscript))

for n_fold, (train_ind, test_ind) in enumerate(logo.split(features, labels, groups=sites)):

    if n_fold < 5: # not testing on site 6 (HC site (Dataset 2 in manuscript))
        print('Processing the No.%i cross-validation in %i-fold CV' % (n_fold + 1, n_sites - 1))
        x_train, y_train = features[train_ind], labels[train_ind]
        x_test, y_test = features[test_ind], labels[test_ind]

        init_clf = SVC(kernel='linear')
        grid = GridSearchCV(init_clf, {'C': C_val}, cv=nested_skf, scoring='balanced_accuracy', n_jobs=n_jobs)
        grid.fit(x_train, y_train)
        print('    The best parameter C: %.2e with BAC of %f' % (grid.best_params_['C'], grid.best_score_))
        clf = SVC(kernel='linear', C=grid.best_params_['C'])
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
eval_df.index = ['CV_' + str(i + 1) for i in range(n_sites - 1)]
print(eval_df)
print('\nAverage Sensitivity: %.4f±%.4f' % (eval_metrics[:, 0].mean(), eval_metrics[:, 0].std()))
print('Average Specificity: %.4f±%.4f' % (eval_metrics[:, 1].mean(), eval_metrics[:, 1].std()))
print('Average balanced accuracy: %.4f±%.4f' % (eval_metrics[:, 2].mean(), eval_metrics[:, 2].std()))
