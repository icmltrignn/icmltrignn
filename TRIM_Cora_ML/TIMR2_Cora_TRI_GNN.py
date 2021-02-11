import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from spektral_tri_gnn.datasets import citation
from spektral_tri_gnn.layers import FGSConv
from spektral_tri_gnn.utils import normalized_laplacian, rescale_laplacian, normalized_adjacency, degree_power
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# Load data #
dataset = 'cora'
adj, node_features, y_train, y_val, y_test, train_mask, val_mask, test_mask = citation.load_data(dataset)
np_adj = adj.toarray()
original_adj = np_adj.copy()


cora_wasserstein_distance_dataset = pd.read_csv("cora_wasserstein_distance_mat_2_nKNN.csv", index_col= 0)
cora_wasserstein_distance = cora_wasserstein_distance_dataset.values # min: 0.5 and max: 625.5

wasserstein_dis_adj = np_adj # assign original adj to wasserstein distance adj firstly for construction

# add edges #
added_threshold = 0
added_edge_x, added_edge_y = np.where((cora_wasserstein_distance > 0) & (cora_wasserstein_distance < added_threshold))
# remove edges #
removed_threshold = 200
removed_edge_x, removed_edge_y = np.where(cora_wasserstein_distance > removed_threshold)
# reconstruction #
wasserstein_dis_adj[added_edge_x, added_edge_y] = wasserstein_dis_adj[added_edge_x, added_edge_y] + 1
wasserstein_dis_adj[removed_edge_x, removed_edge_y] = wasserstein_dis_adj[removed_edge_x, removed_edge_y] - 1
wasserstein_dis_adj = np.clip(wasserstein_dis_adj, a_min= 0, a_max= 1)

wasserstein_dis_adj = sp.csr_matrix(wasserstein_dis_adj) # sparse wasserstein distance adjacency

# VAR #
def var_preprocess(adj, r):
    adj_ = adj + sp.eye(adj.shape[0])
    adj_ = adj_ ** r
    adj_[adj_ > 1] = 1
    adj_  = adj_.toarray()
    adj_ = csr_matrix(adj_)
    rowsum = adj_.sum(1).A1
    degree_mat_inv_sqrt_l = sp.diags(np.power(rowsum, -0.))
    degree_mat_inv_sqrt_r = sp.diags(np.power(rowsum, -1.))
    adj_normalized = adj_.dot(degree_mat_inv_sqrt_l).T.dot(degree_mat_inv_sqrt_r).tocsr()
    return adj_normalized

fltr = var_preprocess(adj=wasserstein_dis_adj, r=2) #normalized_adjacency(wasserstein_dis_adj, symmetric=True)#var_preprocess(adj=wasserstein_dis_adj, r=2)

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

# Parameters #
recurrent = None
N = node_features.shape[0]
F = node_features.shape[1]
n_classes = y_train.shape[1]
dropout_rate = 0.75
l2_reg = 5e-4
learning_rate = 1e-2
epochs = 2000
es_patience = 300
recur_num = 3 

# original filtr is based on original adjacency matrix #
original_adj = csr_matrix(original_adj)
ori_fltr = normalized_adjacency(original_adj, symmetric=True)

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
