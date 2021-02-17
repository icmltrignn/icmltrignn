import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from spektral_tri_gnn.datasets import citation
from spektral_tri_gnn.layers import FGSConv
from spektral_tri_gnn.utils import normalized_laplacian, rescale_laplacian, normalized_adjacency, degree_power
from sklearn.model_selection import GridSearchCV 
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg.eigen.arpack import eigsh
import pandas as pd
import numpy as np
from numpy import linalg
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import inspect
import networkx as nx
import operator
import collections
import matplotlib.pyplot as plt

path = os.getcwd()

# Load data
# adj processing #
adj = pd.read_csv(path + "/ACTIVSg200_adj.csv",index_col=0)
adj = adj.values
adj = csr_matrix(adj)
A_darray = adj.toarray()
transposed_A_darray = np.transpose(A_darray)
sym_A_darray = A_darray + np.multiply(transposed_A_darray,(transposed_A_darray > A_darray)) - np.multiply(A_darray, (transposed_A_darray > A_darray))
adj = csr_matrix(sym_A_darray)

# features #
bus_input = pd.read_csv(path + "/ACTIVSg200_feature_matrix.csv",index_col = 0)
node_features = bus_input.iloc[:,0:5]
node_features = csr_matrix(node_features)

# 3, label #
z = pd.read_csv(path + "/ACTIVSg200_y.csv",index_col = 0).values
z = z.reshape((200,))
idx = np.arange(len(z))
label = z
label_mat = np.zeros((200,3),dtype = float)
for i in range(200):
    label_mat[i,z[i]-1] = 1.

idx_train = pd.read_csv(path +"/ACTIVSg200_idx_train.csv",header=0, index_col=0).values
idx_train = idx_train.reshape(-1,)

idx_val = pd.read_csv(path +"/ACTIVSg200_idx_val.csv",header=0, index_col=0).values
idx_val = idx_val.reshape(-1,)

idx_test = pd.read_csv(path +"/ACTIVSg200_idx_test.csv",header=0, index_col=0).values
idx_test = idx_test.reshape(-1,)

y_train = np.zeros((len(label), 3),dtype= float)
for i in idx_train:
        y_train[i, label[i] - 1] = 1.

y_val = np.zeros((len(label), 3), dtype= float)
for i in idx_val:
        y_val[i, label[i] - 1] = 1.

y_test = np.zeros((len(label), 3), dtype= float)
for i in idx_test:
        y_test[i, label[i] - 1] = 1.

train_mask = np.full((len(label),), False, dtype= bool)
val_mask = np.full((len(label),), False, dtype= bool)
test_mask = np.full((len(label),), False, dtype= bool)
train_mask[idx_train] = True
val_mask[idx_val] = True
test_mask[idx_test] = True

np_adj = adj.toarray()
original_adj = np_adj.copy()


#-------------------------------------------------------------------------------------------------------------------------#
wasserstein_dis_adj = np.zeros(shape= (200,200),dtype= np.float32)
binary_np_adj_x, binary_np_adj_y = np.where(original_adj > 0)
wasserstein_dis_adj[binary_np_adj_x, binary_np_adj_y] = 1. # assign original adj to wasserstein distance adj firstly for construction

il_wasserstein_distance_dataset = pd.read_csv(path + "/ACTIVSg200_wasserstein_distance.csv", index_col= 0)
il_wasserstein_distance = il_wasserstein_distance_dataset.values # min: 0.5; max: 16; mean: 6.7441097957288765

# add edges #
added_threshold = 1
added_edge_x, added_edge_y = np.where((il_wasserstein_distance > 0) & (il_wasserstein_distance < added_threshold))

# remove edges #
removed_threshold = 3
removed_edge_x, removed_edge_y = np.where(il_wasserstein_distance > removed_threshold)

# reconstruction #
wasserstein_dis_adj[added_edge_x, added_edge_y] = wasserstein_dis_adj[added_edge_x, added_edge_y] + 1
wasserstein_dis_adj[removed_edge_x, removed_edge_y] = wasserstein_dis_adj[removed_edge_x, removed_edge_y] - 1
wasserstein_dis_adj = np.clip(wasserstein_dis_adj, a_min= 0, a_max= 1)

wasserstein_dis_adj = sp.csr_matrix(wasserstein_dis_adj) # sparse wasserstein distance adjacency
#-------------------------------------------------------------------------------------------------------------------------#
def var_preprocess(adj, r):
    if isinstance(r, int):
        adj_ = adj + sp.eye(adj.shape[0])
        adj_ = adj_ ** r
        adj_[adj_ > 1] = 1
        adj_ = adj_.toarray()
        adj_ = csr_matrix(adj_)
        rowsum = adj_.sum(1).A1
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5))
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).T.dot(degree_mat_inv_sqrt).tocsr()
    else:
        #adj = adj + sp.eye(adj.shape[0])
        degrees_left = np.float_power(np.array(adj.sum(1)), -r).flatten()
        degrees_left[np.isinf(degrees_left)] = 0.
        normalized_D = sp.diags(degrees_left, 0)
        degrees_right = np.float_power(np.array(adj.sum(1)), (r - 1)).flatten()
        degrees_right[np.isinf(degrees_right)] = 0.
        normalized_D_right = sp.diags(degrees_right, 0)
        adj_normalized = normalized_D.dot(adj)
        adj_normalized = adj_normalized.dot(normalized_D_right)
    return adj_normalized

fltr = var_preprocess(adj=wasserstein_dis_adj, r=0.5)

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
topo_features = (topo_normalization(il_wasserstein_distance).dot(node_features_arr) + node_features_arr)/2.
features = sp.csr_matrix(topo_features)

# Parameters #
recurrent = None
N = node_features.shape[0]
F = node_features.shape[1]
n_classes = y_train.shape[1]
dropout_rate = 0.
l2_reg = 5e-3
learning_rate = 0.1#1e-1
epochs = 500
es_patience = 500
recur_num = 1

original_adj = csr_matrix(original_adj)
ori_fltr = normalized_adjacency(original_adj, symmetric=True)


ori_fltr = fltr

# Model definition #
X_in = Input(shape=(F, ))
fltr_in = Input((N, ), sparse=True)

dropout_1 = Dropout(dropout_rate)(X_in)
graph_conv_1 = FGSConv(32,
                       num_comp=2,
                       num_filter=1,
                       recurrent=recurrent,
                       recur_num = recur_num,
                       dropout_rate=dropout_rate,
                       activation='elu',
                       gcn_activation='elu',
                       kernel_regularizer=l2(l2_reg))([dropout_1, fltr_in])


dropout_2 = Dropout(dropout_rate)(graph_conv_1)
graph_conv_2 = FGSConv(n_classes,
                       num_comp=1,
                       num_filter=1,
                       recurrent=recurrent,
                       recur_num = recur_num,
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

# Train model #
validation_data = ([node_features, ori_fltr], y_val, val_mask)


model.fit([node_features, fltr],
          y_train,
          sample_weight=train_mask,
          epochs=epochs,
          batch_size=N,
          validation_data=validation_data,
          shuffle=False,
          callbacks=callbacks)

# Load best model #
model.load_weights('best_model.h5')

# Evaluate model #
print('Evaluating model.')
eval_results = model.evaluate([node_features, ori_fltr],
                              y_test,
                              sample_weight=test_mask,
                              batch_size=N)
print('Done.\n'
      'Test loss: {}\n'
      'Test accuracy: {}'.format(*eval_results))
