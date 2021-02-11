import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from spektral_fix.datasets import citation
from spektral_fix.layers import FGSConv
from spektral_fix.utils import normalized_laplacian, rescale_laplacian, normalized_adjacency, degree_power
from sklearn.model_selection import GridSearchCV
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg.eigen.arpack import eigsh
import pandas as pd
import numpy as np
from numpy import linalg
from scipy import sparse, io
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import inspect
import networkx as nx
import operator


# Load data
# 1, adj processing #
adj = pd.read_csv("bus_adj_matrix.csv")
adj = adj.values
adj = csr_matrix(adj)
# over #

# 2, features #
bus_input = pd.read_csv("bus_input.csv")
node_features = bus_input.iloc[:,1:3]
node_features = csr_matrix(node_features)
# over #

# 3, label #
label = bus_input.iloc[:,3].values
label_mat = np.zeros((len(label), 3), dtype=float)

idx = pd.read_csv("idx.csv").values
idx = idx.reshape(-1,)

idx_train = pd.read_csv("idx_train.csv").values
idx_train = idx_train.reshape(-1,)


idx_val = pd.read_csv("idx_val.csv").values
idx_val = idx_val.reshape(-1,)

idx_test = pd.read_csv("idx_test.csv").values
idx_test = idx_test.reshape(-1,)


y_train = np.zeros((len(label), 3),dtype= float)
for i in idx_train:
    if label[i] == 4:
        y_train[i, label[i] - 2] = 1.
    else:
        y_train[i, label[i] - 1] = 1.

y_val = np.zeros((len(label), 3), dtype= float)
for i in idx_val:
    if label[i] == 4:
        y_val[i, label[i] - 2] = 1.
    else:
        y_val[i, label[i] - 1] = 1.

y_test = np.zeros((len(label), 3), dtype= float)
for i in idx_test:
    if label[i] == 4:
        y_test[i, label[i] - 2] = 1.
    else:
        y_test[i, label[i] - 1] = 1.

train_mask = np.full((len(label),), False, dtype= bool)
val_mask = np.full((len(label),), False, dtype= bool)
test_mask = np.full((len(label),), False, dtype= bool)
train_mask[idx_train] = True
val_mask[idx_val] = True
test_mask[idx_test] = True

A = adj
A_darray = A.toarray()
transposed_A_darray = np.transpose(A_darray)
sym_A_darray = A_darray + np.multiply(transposed_A_darray,(transposed_A_darray > A_darray)) - np.multiply(A_darray, (transposed_A_darray > A_darray))
A = csr_matrix(sym_A_darray)
adj = A

original_adj = adj.copy()
np_adj = adj.toarray()

# TIMR 2 #
ieee_wasserstein_distance_dataset = pd.read_csv("ieee_wasserstein_distance_mat_5_nKNN.csv", index_col= 0)
ieee_wasserstein_distance = ieee_wasserstein_distance_dataset.values # min: 1.0; max: 60.0; mean: 19.477704678362574

wasserstein_dis_adj = np_adj
# add edges #
added_threshold = 0
added_edge_x, added_edge_y = np.where((ieee_wasserstein_distance > 0) & (ieee_wasserstein_distance <= added_threshold))

# remove edges #
removed_threshold = 25
removed_edge_x, removed_edge_y = np.where(ieee_wasserstein_distance > removed_threshold)

# reconstruction #
wasserstein_dis_adj[added_edge_x, added_edge_y] = wasserstein_dis_adj[added_edge_x, added_edge_y] + 1
wasserstein_dis_adj[removed_edge_x, removed_edge_y] = wasserstein_dis_adj[removed_edge_x, removed_edge_y] - 1
wasserstein_dis_adj = np.clip(wasserstein_dis_adj, a_min= 0, a_max= 1)

wasserstein_dis_adj = sp.csr_matrix(wasserstein_dis_adj) # sparse wasserstein distance adjacency

# STAN #
def topo_normalization(x):
    # diagonal part #
    diagonal_mask = np.eye(x.shape[0], dtype= bool)
    x[diagonal_mask] = 1. # count self's attributes
    # non-diagonal part convert zero to one #
    x[np.where(x==0)] = 1.
    inv_x = 1./x
    # normalize #
    exp_inv_x = np.exp(inv_x)/np.sum(np.exp(inv_x),axis=1)
    return exp_inv_x

node_features_arr = node_features.toarray()
topo_features = (topo_normalization(ieee_wasserstein_distance).dot(node_features_arr) + node_features_arr)/2.
features = sp.csr_matrix(topo_features)

adj = wasserstein_dis_adj
original_adj = csr_matrix(original_adj)


# Parameters
ARMA_K = 7
ARMA_D = 1
recurrent = None
N = node_features.shape[0]
F = node_features.shape[1]
n_classes = y_train.shape[1]
dropout_rate = 0
l2_reg = 5e-4
learning_rate = 1e-1
epochs = 500
es_patience = 300
recur_num = 3


# first Fractional-G-SSL part #
gamma = 0.005
degrees = np.array(adj.sum(1)).flatten()
degrees[np.isinf(degrees)] = 0.
D = sp.diags(degrees, 0)
L = D - adj


L_darray = L.toarray()
D, V = np.linalg.eigh(L_darray, 'U')
M_gamma_Lambda = D
M_gamma_Lambda[M_gamma_Lambda < 1e-10] = 0
M_V = V

M_gamma_Lambda = np.float_power(M_gamma_Lambda, gamma)
M_gamma_Lambda = np.diag(M_gamma_Lambda, 0)
M_gamma_Lambda = sp.csr_matrix(M_gamma_Lambda)
M_V = sp.csr_matrix(M_V)
Lg = M_V * M_gamma_Lambda
Lg = Lg * sp.csr_matrix.transpose(M_V)

Lg = Lg.toarray()
Lg = Lg.reshape(1, -1)
Lg[abs(Lg) < 1e-10] = 0.
Lg = Lg.reshape(N, -1)
Dg = np.diag(np.diag(Lg))
Ag = Dg - Lg
Ag = sp.csr_matrix(Ag)

alpha = 0.2
power_Dg = np.float_power(np.diag(Dg), -alpha)
power_Dg = np.diag(power_Dg)
power_Dg = sp.csr_matrix(power_Dg)

power_Dg_right = np.float_power(np.diag(Dg), (alpha - 1))
power_Dg_right = np.diag(power_Dg_right)
power_Dg_right = sp.csr_matrix(power_Dg_right)

fltr = power_Dg * Ag
fltr = fltr * power_Dg_right

# second Fractional-G-SSL part #
gamma = 0.005
degrees = np.array(original_adj.sum(1)).flatten()
degrees[np.isinf(degrees)] = 0.
D = sp.diags(degrees, 0)
L = D - original_adj


L_darray = L.toarray()
D, V = np.linalg.eigh(L_darray, 'U')
M_gamma_Lambda = D
M_gamma_Lambda[M_gamma_Lambda < 1e-10] = 0
M_V = V

M_gamma_Lambda = np.float_power(M_gamma_Lambda, gamma)
M_gamma_Lambda = np.diag(M_gamma_Lambda, 0)
M_gamma_Lambda = sp.csr_matrix(M_gamma_Lambda)
M_V = sp.csr_matrix(M_V)
Lg = M_V * M_gamma_Lambda
Lg = Lg * sp.csr_matrix.transpose(M_V)

Lg = Lg.toarray()
Lg = Lg.reshape(1, -1)
Lg[abs(Lg) < 1e-10] = 0.
Lg = Lg.reshape(N, -1)
Dg = np.diag(np.diag(Lg))
Ag = Dg - Lg
Ag = sp.csr_matrix(Ag)

alpha = 0.2
power_Dg = np.float_power(np.diag(Dg), -alpha)
power_Dg = np.diag(power_Dg)
power_Dg = sp.csr_matrix(power_Dg)

power_Dg_right = np.float_power(np.diag(Dg), (alpha - 1))
power_Dg_right = np.diag(power_Dg_right)
power_Dg_right = sp.csr_matrix(power_Dg_right)

ori_fltr = power_Dg * Ag
ori_fltr = ori_fltr * power_Dg_right

# Model definition
X_in = Input(shape=(F,))
fltr_in = Input((N,), sparse=True)

dropout_1 = Dropout(dropout_rate)(X_in)
graph_conv_1 = FGSConv(128,
                        ARMA_K=ARMA_K,
                        ARMA_D=ARMA_D,
                        recurrent=recurrent,
                        recur_num=recur_num,
                        dropout_rate=dropout_rate,
                        activation='elu',
                        gcn_activation='elu',
                        kernel_regularizer=l2(l2_reg))([dropout_1, fltr_in])

dropout_2 = Dropout(dropout_rate)(graph_conv_1)
graph_conv_2 = FGSConv(n_classes,
                        ARMA_K=2,
                        ARMA_D=1,
                        recurrent=recurrent,
                        recur_num=recur_num,
                        dropout_rate=dropout_rate,
                        activation='softmax',
                        gcn_activation=None,
                        kernel_regularizer=l2(l2_reg))([dropout_2, fltr_in])

# Build model
model = Model(inputs=[X_in, fltr_in], outputs=graph_conv_2)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])
model.summary()



callbacks = [
    EarlyStopping(monitor='val_weighted_acc', patience=es_patience),
    ModelCheckpoint('best_model.h5', monitor='val_weighted_acc',
                    save_best_only=True, save_weights_only=True)
]

# Train model
validation_data = ([node_features, ori_fltr], y_val, val_mask)



model.fit([node_features, fltr],
          y_train,
          sample_weight=train_mask,
          epochs=epochs,
          batch_size=N,
          validation_data=validation_data,
          shuffle=False,
          callbacks=callbacks)

# Load best model
model.load_weights('best_model.h5')

# Evaluate model
print('Evaluating model.')
eval_results = model.evaluate([node_features, ori_fltr],
                              y_test,
                              sample_weight=test_mask,
                              batch_size=N)
print('Done.\n'
      'Test loss: {}\n'
      'Test accuracy: {}'.format(*eval_results))
