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

# Load data #
dataset = 'citeseer'
adj, node_features, y_train, y_val, y_test, train_mask, val_mask, test_mask = citation.load_data(dataset)
np_adj = adj.toarray()
original_adj = adj.copy()

def topo_softmax(x):
    mask = ((x>0) -1) * 999999
    x_1 = np.exp(x + mask)
    x_2 = np.sum(x_1, axis=1)
    x_3 = x_1 / x_2.reshape(-1,1)
    nan_mask = np.isnan(x_3)
    x_3[nan_mask] = 0.
    return x_3

topo_var_2_dis_dgm_k_2 = np.load(path + '/inverse_wasserstein_distance.npy.npy', allow_pickle= True)


# VAR #
pos_x_labels, pos_y_labels = np.where(topo_var_2_dis_dgm_k_2>1.)
percent_pos = 1.
sample_label_pos = (np.random.permutation(range(pos_x_labels.shape[0])))[:int(pos_x_labels.shape[0] * percent_pos)]
preserve_pos_x_labels = pos_x_labels[sample_label_pos]
preserve_pos_y_labels = pos_x_labels[sample_label_pos]

neg_x_labels, neg_y_labels = np.where(((topo_var_2_dis_dgm_k_2>0)&(topo_var_2_dis_dgm_k_2<0.1)))
percent_neg = 0.001
sample_label_neg = (np.random.permutation(range(neg_x_labels.shape[0])))[:int(neg_x_labels.shape[0] * percent_neg)]
preserve_neg_x_labels = neg_x_labels[sample_label_neg]
preserve_neg_y_labels = neg_y_labels[sample_label_neg]


def var_preprocess(adj, r, pos_x_labels, pos_y_labels, neg_x_labels, neg_y_labels):
    if isinstance(r, int):
        adj_ = adj + sp.eye(adj.shape[0])
        adj_ = adj_ ** r
        adj_[adj_ > 1] = 1
        adj_ = adj_.toarray()
        adj_[pos_x_labels, pos_y_labels] = 1
        adj_[neg_x_labels, neg_y_labels] = 0
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

fltr = var_preprocess(adj=adj, r=0.001, pos_x_labels = preserve_pos_x_labels, pos_y_labels= preserve_pos_y_labels, neg_x_labels= preserve_neg_x_labels
                                 , neg_y_labels=preserve_neg_y_labels)


# STAN #
# topo attension mechanism #
ec_x, ec_y = np.where(np.triu(topo_var_2_dis_dgm_k_2, k = 1)> 2.)

# filter topo distance #
topo_var_2_dis_dgm_k_2[ec_x,ec_y] = 2.
topo_var_2_dis_dgm_k_2[ec_y,ec_x] = 2.

norm_topo_var_2_dis_dgm_k_2 = topo_softmax(topo_var_2_dis_dgm_k_2)

# scalar setting #
weight_1 = 500.
weight_2 = 1.
norm_topo_var_2_dis_dgm_k_2 = norm_topo_var_2_dis_dgm_k_2 + np.eye(norm_topo_var_2_dis_dgm_k_2.shape[0])
new_features = (np.matmul(norm_topo_var_2_dis_dgm_k_2, node_features.toarray())/weight_1 + node_features.toarray())/weight_2
node_features = sp.lil_matrix(new_features)


# Parameters #
recurrent = None
N = node_features.shape[0]
F = node_features.shape[1]
n_classes = y_train.shape[1]
dropout_rate = 0.75
l2_reg = 5e-4
learning_rate = 1e-2
epochs = 2000
es_patience = 150
recur_num = 3 

sigma = 0.55
ori_degrees = np.float_power(np.array(original_adj.sum(1)), -sigma).flatten()
ori_degrees[np.isinf(ori_degrees)] = 0.
ori_normalized_D = sp.diags(ori_degrees, 0)

ori_degrees_sec = np.float_power(np.array(original_adj.sum(1)), (sigma - 1)).flatten()
ori_degrees_sec[np.isinf(ori_degrees_sec)] = 0.
ori_normalized_D_sec = sp.diags(ori_degrees_sec, 0)
ori_fltr = ori_normalized_D.dot(original_adj)
ori_fltr = ori_fltr.dot(ori_normalized_D_sec)



# Model definition #
X_in = Input(shape=(F, ))
fltr_in = Input((N, ), sparse=True)

dropout_1 = Dropout(dropout_rate)(X_in)
graph_conv_1 = FGSConv(32,
                       num_comp=4,
                       num_filter=1,
                       recurrent=recurrent,
                       recur_num = recur_num,
                       dropout_rate=dropout_rate,
                       activation='elu',
                       gcn_activation='elu',
                       kernel_regularizer=l2(l2_reg))([dropout_1, fltr_in])


dropout_2 = Dropout(dropout_rate)(graph_conv_1)
graph_conv_2 = FGSConv(n_classes,
                       num_comp=2,
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
