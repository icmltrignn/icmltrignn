from __future__ import absolute_import

from keras import activations, initializers, regularizers, constraints
from keras import backend as K
from keras.layers import Layer, LeakyReLU, Dropout, AveragePooling2D, AveragePooling1D, MaxPooling1D, MaxPooling2D, Conv1D, BatchNormalization, Conv2D, Flatten, GlobalMaxPooling1D
from keras.constraints import max_norm, non_neg, unit_norm, min_max_norm
from keras.initializers import RandomUniform
import tensorflow as tf
import numpy as np


_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

class FGSConv(Layer):

    def __init__(self,
                 channels,
                 ARMA_D,
                 ARMA_K=None,
                 recurrent=None,
                 recur_num = None,
                 gcn_activation='relu',
                 dropout_rate=0.0,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(FGSConv, self).__init__(**kwargs)
        self.channels = channels # YC: channels means neuros #
        self.ARMA_D = ARMA_D
        self.ARMA_K = ARMA_D if ARMA_K is None else ARMA_K
        self.recurrent = recurrent
        self.recur_num = recur_num
        self.activation = activations.get(activation)
        self.gcn_activation = activations.get(gcn_activation)
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = False

    def build(self, input_shape):
        assert len(input_shape) >= 2
        # When using shared weights, pre-compute them here
        if self.recurrent is None:
            self.kernels_in = []  # Weights from input space to output space
            self.kernels_hid = []  # Weights from output space to output space
            for k in range(self.ARMA_K):
                self.kernels_in.append(self.get_gcn_weights(input_shape[0][-1],# input_shape[0][1]
                                                            input_shape[0][-1],
                                                            self.channels,
                                                            name='ARMA_skip_{}r_in'.format(k),
                                                            use_bias=self.use_bias,
                                                            recur_num = self.recur_num,
                                                            kernel_initializer=self.kernel_initializer,
                                                            bias_initializer=self.bias_initializer,
                                                            kernel_regularizer=self.kernel_regularizer,
                                                            bias_regularizer=self.bias_regularizer,
                                                            kernel_constraint=self.kernel_constraint,
                                                            bias_constraint=self.bias_constraint))
                if self.ARMA_D > 1:
                    for d in range(self.ARMA_D-1):
                        self.kernels_hid.append(self.get_gcn_weights(self.channels,
                                                                     input_shape[0][-1],
                                                                     self.channels,
                                                                     name='ARMA_skip_{}r_hid'.format(
                                                                         k * len(range(self.ARMA_D-1)) + d),
                                                                     use_bias=self.use_bias,
                                                                     recur_num=self.recur_num,
                                                                     kernel_initializer=self.kernel_initializer,
                                                                     bias_initializer=self.bias_initializer,
                                                                     kernel_regularizer=self.kernel_regularizer,
                                                                     bias_regularizer=self.bias_regularizer,
                                                                     kernel_constraint=self.kernel_constraint,
                                                                     bias_constraint=self.bias_constraint))


        self.built = True

    def call(self, inputs):
        features = inputs[0]
        fltr = inputs[1]

        # Convolution
        output = []  # Stores the parallel filters
        for k in range(self.ARMA_K):
            output_k = features
            for dd in range(self.ARMA_D):
                features_drop = Dropout(self.dropout_rate)(features)

                output_k = self.graph_conv_skip([output_k, features_drop, fltr],
                                                self.channels,
                                                recurrent_k=k,
                                                recurrent_d=dd,
                                                activation=self.gcn_activation,
                                                use_bias=self.use_bias,
                                                recur_num = self.recur_num,
                                                kernel_initializer=self.kernel_initializer,
                                                bias_initializer=self.bias_initializer,
                                                kernel_regularizer=self.kernel_regularizer,
                                                bias_regularizer=self.bias_regularizer,
                                                kernel_constraint=self.kernel_constraint,
                                                bias_constraint=self.bias_constraint)
            output.append(output_k)



        output = K.concatenate(output, axis=-1)


        # Average pooling
        output = K.expand_dims(output, axis=-1)
        output_dim = K.int_shape(output)



        if len(output_dim) == 3:

            if self.channels != 3:
                output = self.gated_max_avg_pooling(pooling_input=output)

            elif self.channels == 3:
                output = self.gated_max_avg_pooling(pooling_input=output)

        elif len(output_dim) == 4:
            output = tf.reshape(output, [-1,self.channels,self.ARMA_K,1])
            output = AveragePooling2D(pool_size=(1, self.ARMA_K), padding='same')(output)
            output = tf.reshape(output, [-1,self.channels,1])
        else:
            raise RuntimeError('GCN_ARMA layer: wrong output dim')
        output = K.squeeze(output, axis=-1)

        if self.channels !=3:

            output = tf.nn.elu(output) + output
            output = tf.nn.elu(output)

        elif self.channels == 3:
            output = tf.nn.softmax(output)

        return output

    def compute_output_shape(self, input_shape):
        features_shape = input_shape[0]
        if self.channels!= 3:
            output_shape = features_shape[:-1] + (self.channels,)
        elif self.channels == 3:
            output_shape = features_shape[:-1] + (self.channels,)
        return output_shape

    def get_config(self):
        config = {
            'channels': self.channels,
            'ARMA_D': self.ARMA_D,
            'ARMA_K': self.ARMA_K,
            'recurrent': self.recurrent,
            'gcn_activation': activations.serialize(self.gcn_activation),
            'dropout_rate': self.dropout_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(FGSConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_gcn_weights(self, input_dim, input_dim_skip, channels, name,
                        recur_num,
                        use_bias=True,
                        kernel_initializer='glorot_uniform',
                        bias_initializer='zeros',
                        kernel_regularizer=None,
                        bias_regularizer=None,
                        kernel_constraint=None,
                        bias_constraint=None):

        kernel_initializer = initializers.get(kernel_initializer)
        kernel_regularizer = regularizers.get(kernel_regularizer)
        kernel_constraint = constraints.get(kernel_constraint)
        bias_initializer = initializers.get(bias_initializer)
        bias_regularizer = regularizers.get(bias_regularizer)
        bias_constraint = constraints.get(bias_constraint)

        kernel_1 = self.add_weight(shape=(input_dim, channels),
                                   name=name + '_kernel_1',
                                   initializer=kernel_initializer,
                                   regularizer=kernel_regularizer,
                                   constraint=kernel_constraint)


        if use_bias:
            bias = self.add_weight(shape=(channels,),
                                   name=name + '_bias',
                                   initializer=bias_initializer,
                                   regularizer=bias_regularizer,
                                   constraint=bias_constraint)
        else:
            bias = None
        return kernel_1, bias

    # ------------------------------------------------- #
    # gated-max-average pooling layer #
    def gated_max_avg_pooling(self,
                              pooling_input):

        transformed_pooling_input = tf.reshape(pooling_input, shape=[-1, 1])
        alpha_kernel = self.get_pooling_weight(input_shape=self.ARMA_K * self.channels)
        sigmoid_alpha_kernel = tf.matmul(alpha_kernel, transformed_pooling_input)
        sigmoid_alpha_kernel = tf.sigmoid(sigmoid_alpha_kernel)

        x1 = AveragePooling1D(pool_size=self.ARMA_K, padding='same')(pooling_input)
        x2 = MaxPooling1D(pool_size=self.ARMA_K, padding='same')(pooling_input)
        temp_output = tf.add(tf.multiply(x1, sigmoid_alpha_kernel), tf.multiply(x2, (1 - sigmoid_alpha_kernel)))
        output = temp_output

        return output

    # max-average pooling layer #
    def max_avg_pooling(self,
                        pooling_input):

        alpha_kernel = self.alpha_weight()

        x1 = AveragePooling1D(pool_size=self.ARMA_K, padding='same')(pooling_input)
        x2 = MaxPooling1D(pool_size=self.ARMA_K, padding='same')(pooling_input)
        temp_output = tf.add(tf.multiply(x1, alpha_kernel), tf.multiply(x2, (1 - alpha_kernel)))
        output = temp_output


        return output

    def get_pooling_weight(self, input_shape,
                           kernel_initializer='glorot_uniform',
                           kernel_regularizer=None,
                           kernel_constraint=None):
        layer = self.__class__.__name__.lower()
        pooling_kernel = self.add_weight(shape=(1, input_shape * 118),
                                         name='max_avg_pooling_kernel' + '_' + layer + '_' + str(
                                             get_layer_uid(layer)),
                                         initializer=kernel_initializer,
                                         regularizer=kernel_regularizer,
                                         constraint=kernel_constraint,
                                         trainable=True)
        return pooling_kernel

    # the weight variable for pooling kernel #
    def alpha_weight(self,
                     kernel_initializer=RandomUniform(minval=0.1, maxval=0.9),
                     kernel_regularizer=None,
                     kernel_constraint=min_max_norm(min_value=0.09, max_value=1)):
        layer = self.__class__.__name__.lower()
        pooling_alpha = self.add_weight(shape=(1,),
                                        name='alpha_pooling_kernel' + '_' + layer + '_' + str(
                                            get_layer_uid(layer)),
                                        initializer=kernel_initializer,
                                        regularizer=kernel_regularizer,
                                        constraint=kernel_constraint,
                                        trainable=True)
        return pooling_alpha

    # the weight variable for arma graph filter kernel #
    def beta_weight(self,
                    kernel_initializer=RandomUniform(minval=0.1, maxval=0.9),
                    kernel_regularizer=None,
                    kernel_constraint=min_max_norm(min_value=0.09, max_value=1)):
        layer = self.__class__.__name__.lower()
        arma_beta = []
        for _ in range(2):
            arma_beta.append(self.add_weight(shape=(1,),
                                             name='arma_beta_kernel' + '_' + layer + '_' + str(
                                                 get_layer_uid(layer)),
                                             initializer=kernel_initializer,
                                             regularizer=kernel_regularizer,
                                             constraint=kernel_constraint,
                                             trainable=True))
        return arma_beta

    # the weight variables (list output) for hybrid pooling kernels #
    # tune the number of \x you want to use hybrid weight, do not set the overwhelming hybrid weights #
    def hybrid_pooling_weights(self,
                               kernel_initializer = RandomUniform(minval=0.499, maxval=0.501),
                               kernel_regularizer = None,
                               kernel_constraint = min_max_norm(min_value = 0.495, max_value = 0.505)):
        hybrid_weights = [[] for i in range(2)]
        for mm in range(2):
            for kk in range(self.channels):
                layer = self.__class__.__name__.lower()
                hybrid_kernel = self.add_weight(shape=(1,),
                                                name='hybrid_pooling_kernel' + layer + '_' + str(
                                                    get_layer_uid()) + '_' + str(kk),
                                                initializer=kernel_initializer,
                                                regularizer=kernel_regularizer,
                                                constraint=kernel_constraint,
                                                trainable=True
                                                )
                hybrid_weights[mm].append(hybrid_kernel)
        return hybrid_weights

    def reduce_median(self, input):
        if self.ARMA_K % 2 == 0:
            median_output = tf.reduce_mean(input, axis=1)
        else:
            temp_k = int(np.floor(self.ARMA_K / 2))
            temp_top_k = tf.nn.top_k(input, k=temp_k).values
            median_output = temp_top_k[:, 0]
        return median_output


    def median_avg_pooling(self,
                        pooling_input):

        alpha_kernel = self.alpha_weight()

        road_output = tf.squeeze(pooling_input, axis=-1)
        temp = self.reduce_median(road_output[:, slice(0, self.ARMA_K)])
        temp = K.expand_dims(temp, axis=-1)
        for ii in range(1, self.channels):
            temp_inner = road_output[:, slice(ii * self.ARMA_K, ii * self.ARMA_K + self.ARMA_K)]
            mean_temp_inner = self.reduce_median(temp_inner)
            mean_temp_inner = K.expand_dims(mean_temp_inner, axis=-1)
            final_output = tf.concat([temp, mean_temp_inner], axis=1)
            temp = final_output
        x2 = K.expand_dims(final_output, axis=-1)

        x1 = AveragePooling1D(pool_size=self.ARMA_K, padding='same')(pooling_input)
        temp_output = tf.add(tf.multiply(x1, alpha_kernel), tf.multiply(x2, (1 - alpha_kernel)))
        output = temp_output


        return output


    # single scalar version #
    def get_activation_weight(self, input_shape,
                           kernel_initializer='glorot_uniform',
                           kernel_regularizer=None,
                           kernel_constraint=None):
        layer = self.__class__.__name__.lower()
        activation_kernel = self.add_weight(shape=(1, input_shape * 118),
                                         name='activation_kernel' + '_' + layer + '_' + str(
                                             get_layer_uid(layer)),
                                         initializer=kernel_initializer,
                                         regularizer=kernel_regularizer,
                                         constraint=kernel_constraint,
                                         trainable=True)
        return activation_kernel

    # union scalar version #
    def union_activation_weight(self,
                     kernel_initializer=RandomUniform(minval=0.1, maxval=0.9),
                     kernel_regularizer=None,
                     kernel_constraint=min_max_norm(min_value=0.65, max_value=1.00)):
        layer = self.__class__.__name__.lower()
        union_activation_kernel = []
        for tt in range(118):
            union_activation_kernel.append(self.add_weight(shape=(1, ),
                                         name='union_activation_kernel' + '_' + layer + '_' + str(
                                             get_layer_uid(layer)),
                                         initializer=kernel_initializer,
                                         regularizer=kernel_regularizer,
                                         constraint=kernel_constraint,
                                         trainable=True))
        return union_activation_kernel
    # --------------------------------------------------#


    def graph_conv_skip(self, x, channels, #name,
                        recurrent_k=None,
                        recurrent_d=None,
                        activation=None,
                        use_bias=True,
                        recur_num = None,
                        kernel_initializer='glorot_uniform',
                        bias_initializer='zeros',
                        kernel_regularizer=None,
                        bias_regularizer=None,
                        kernel_constraint=None,
                        bias_constraint=None):

        if recurrent_d == 0:
            kernel_1, bias = self.kernels_in[recurrent_k]
        else :
            kernel_1, bias = self.kernels_hid[recurrent_k * (len(range(self.ARMA_D-1))) + recurrent_d -1]
        features = x[0]
        features_skip = x[1]
        fltr = x[2]


        # GSSL #
        alpha = 0.01
        alpha = 1.0 / alpha
        recur_depth = int(np.ceil(4 / alpha))
        normalized_adj = fltr / (alpha)
        new_feature = features
        for _ in range(recur_depth):
            new_feature = K.dot(normalized_adj, new_feature)
            new_feature += features
        new_feature = new_feature - features + features_skip
        new_feature *= (alpha-1) / alpha
        output = K.dot(new_feature, kernel_1)


        if use_bias:
            output = K.bias_add(output, bias)
        if activation is not None:
            output = activations.get(activation)(output)
        return output



def mixed_mode_dot(fltr, features):
    """
    Computes the equivalent of tf.einsum('ij,bjk->bik', fltr, output), but works
    for both dense and sparse fltr.
    :param fltr: rank 2 tensor, the filter for convolution
    :param features: rank 3 tensor, the features of the input signals
    :return:
    """
    _, m_, f_ = K.int_shape(features)
    features = K.permute_dimensions(features, [1, 2, 0])
    features = K.reshape(features, (m_, -1))
    features = K.dot(fltr, features)
    features = K.reshape(features, (m_, f_, -1))
    features = K.permute_dimensions(features, [2, 0, 1])

    return features


def filter_dot(fltr, features):
    """
    Performs the multiplication of a graph filter (N x N) with the node features,
    automatically dealing with single, mixed, and batch modes.
    :param fltr: the graph filter(s) (N x N in single and mixed mode,
    batch x N x N in batch mode).
    :param features: the node features (N x F in single mode, batch x N x F in
    mixed and batch mode).
    :return: the filtered features.
    """
    if len(K.int_shape(features)) == 2:
        # Single mode
        return K.dot(fltr, features)
    else:
        if len(K.int_shape(fltr)) == 3:
            # Batch mode
            return K.batch_dot(fltr, features)
        else:
            # Mixed mode
            return mixed_mode_dot(fltr, features)

def cheby_conv(x0, fltr, recur_num, cheby_weight):
    x = []
    x.append(x0)
    if recur_num > 1:
        x1 = K.dot(fltr, x0)
        x.append(x1)

    for kk in range(2, recur_num):
        x2 = 2 * K.dot(fltr, x1) - x0
        x.append(x2)
        x0, x1 = x1, x2

    pre_res = []

    for ii in range(recur_num):
        pre_res.append(K.dot(x[ii],cheby_weight[ii]))

    out = tf.add_n(pre_res)
    return out
