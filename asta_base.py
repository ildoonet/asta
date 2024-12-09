
from maldi_learn.driams import DRIAMSDatasetExplorer, load_driams_dataset
from maldi_learn.utilities import stratify_by_species_and_label
from maldi_learn.vectorization import BinningVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, average_precision_score

import pandas as pd

DRIAMS_ROOT = '/Users/curtis/Repos/DRIAMS/'


explorer = DRIAMSDatasetExplorer(DRIAMS_ROOT)
antibiotics = 'Ceftriaxone'

driams_dataset = load_driams_dataset(
            explorer.root,
            'DRIAMS-A',
            '*',
            'Escherichia coli',     # e. coli
            [antibiotics],
            # '*',
            # '*',
            # ['2015', '2017'],
            # 'Staphylococcus aureus',
            # ['Ciprofloxacin'],
            handle_missing_resistance_measurements='remove_if_all_missing',
            on_error='warn',
            spectra_type='preprocessed'
)

print(driams_dataset.X[0].shape)
print(driams_dataset.X[1].shape)

bv = BinningVectorizer(2000, 
                    #    min_bin=2000, max_bin=20000
                       )
X = bv.fit_transform(driams_dataset.X)

train_idx, test_idx = stratify_by_species_and_label(driams_dataset.y, antibiotic=antibiotics)

print(driams_dataset.n_samples)

y = driams_dataset.to_numpy(antibiotics)

print(y[train_idx].dtype)
print(X[train_idx].shape)

print(X[0])
print(y[0])
print(X[1])

print('-----')

# remove all idx (train_idx) which contains NaN in X
train_idx = train_idx[~pd.isna(X[train_idx]).any(axis=1)]

lr = LogisticRegression()

lr.fit(X[train_idx], y[train_idx])
y_pred = lr.predict(X[test_idx])

print('-----')
print('Light GBM(Paper)', 0.62)

print('-----')
clf = MLPClassifier(solver='adam', 
                    learning_rate='adaptive',
                    learning_rate_init=0.001,
                    max_iter=5000,
                    batch_size=256,
                    hidden_layer_sizes=(150, 200, 100, 100), random_state=1)
clf = clf.fit(X[train_idx], y[train_idx])

y_pred = clf.predict(X[train_idx])
print('MLP (Train)', average_precision_score(y[train_idx], y_pred, average='macro'))

y_pred = clf.predict(X[test_idx])

# # test: random prediction (0~1), same shape as y_pred
# import numpy as np
# y_pred = np.random.rand(*y_pred.shape)
print('MLP  (Test)', average_precision_score(y[test_idx], y_pred, average='macro'))