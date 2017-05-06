import pandas as pd
import numpy as np
import keras
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Merge
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import TimeDistributed, Lambda
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers.advanced_activations import PReLU
import pickle
from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras.layers import Layer
from keras.models import Model
from keras.layers.core import  Lambda,Dropout,Dense, Flatten, Activation
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import Bidirectional
from keras.layers.convolutional import Conv1D
from keras.layers.recurrent import GRU,LSTM
from keras.layers.pooling import MaxPooling2D
from keras.layers import Concatenate, Input, concatenate,dot
from keras.layers.normalization import BatchNormalization
from keras import initializers as initializations
from keras import regularizers
from keras import constraints
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    :param kwargs:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializations.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = K.dot(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        mul_a = uit  * self.u # with this
        ait = K.sum(mul_a, axis=2) # and this

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[-1]

    def compute_output_shape(self, input_shape):
        """Shape transformation logic so Keras can infer output shape
        """
        return (input_shape[0], input_shape[-1])
def cosine_distance(vests):
    x, y = vests
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0],1)
def time_distr_embedd(model):
    model.add(TimeDistributed(Dense(EMBEDDING_DIM, activation='relu')))
    model.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(EMBEDDING_DIM,)))
def embedd_layer(model):
    model.add(Embedding(embedding_matrix.shape[0],
            EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=MAX_SEQUENCE_LENGTH,
            trainable=False))
def conv_layer(model,drop,dim,num_dense):
    model.add(Convolution1D(filters=dim,
                             kernel_size=filter_length,
                             padding='valid',
                             activation='relu',
                             strides=1))
    model.add(Dropout(drop))

    model.add(Convolution1D(filters=dim,
                             kernel_size=filter_length,
                             padding='valid',
                             activation='relu',
                             strides=1))

    model.add(GlobalMaxPooling1D())
    model.add(Dropout(drop))
    model.add(BatchNormalization())

    model.add(Dense(num_dense))
    model.add(Dropout(drop))
    model.add(BatchNormalization())

def dense_layer(model,num_dense,drop_dense):
    model.add(Dense(num_dense))
    model.add(PReLU())
    model.add(Dropout(drop_dense))
    model.add(BatchNormalization())
def f1_score(y_true,y_pred):
    y = K.eval(y_pred)
    y = [1 if x >.5 else 0 for x in y]
    return metrics.f1_score(y_true,y)

embedding_matrix = pickle.load(open('data/embedd_matrix.p','rb'))
x1 = pickle.load(open('data/q1_train.p','rb'))
x2 = pickle.load(open('data/q2_train.p','rb'))
y = pd.read_csv('data/train.csv').is_duplicate
word_index = embedding_matrix[0]-1
max_features = 200000
filter_length = 5
nb_filter = 64
EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 30
num_lstm = np.random.randint(175, 275)
num_dense = np.random.randint(100, 150)
drop_lstm = 0.15 + np.random.rand() * 0.25
drop_dense = 0.15 + np.random.rand() * 0.25
STAMP = 'lstm_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, drop_lstm, \
        drop_dense)
print('Build model...')

embedd_1 = Sequential(name="Embedd_1")
embedd_layer(embedd_1)
time_distr_embedd(embedd_1)
embedd_2 = Sequential(name ="Embedd_2")
embedd_layer(embedd_2)
time_distr_embedd(embedd_2)

conv_1 = Sequential(name="Conv q_1")
embedd_layer(conv_1)
conv_layer(conv_1,drop_dense,nb_filter,num_dense)
conv_2 = Sequential(name="Conv q_2")
embedd_layer(conv_2)
conv_layer(conv_2,drop_dense,nb_filter,num_dense)

lstm_1 = Sequential(name="LSTM q_1")
embedd_layer(lstm_1)
lstm_1.add(Bidirectional(LSTM(num_lstm, dropout=drop_lstm, recurrent_dropout=drop_lstm)))

lstm_2 = Sequential(name="LSTM q_2")
embedd_layer(lstm_2)
lstm_2.add(Bidirectional(LSTM(num_lstm, dropout=drop_lstm, recurrent_dropout=drop_lstm)))

cos_dist_embedd = Sequential(name="cos_dist_embedd")
cos_dist_embedd.add(Merge([embedd_1,embedd_2],mode='cos'))
cos_dist_embedd.add(Flatten())
cos_dist_lstm = Sequential(name="cos_dist_lstm")
cos_dist_lstm.add(Merge([lstm_1,lstm_2],mode='cos'))
cos_dist_lstm.add(Flatten())
cos_dist_conv = Sequential(name="cos_dist_conv")
cos_dist_conv.add(Merge([conv_1,conv_2],mode='cos'))
cos_dist_conv.add(Flatten())
eucl_dist_embedd = Sequential(name="eucl_dist_embedd")
eucl_dist_embedd.add(Merge([embedd_1,embedd_2],mode=euclidean_distance,output_shape = eucl_dist_output_shape))
eucl_dist_lstm = Sequential(name="eucl_dist_lstm")
eucl_dist_lstm.add(Merge([lstm_1,lstm_2],mode=euclidean_distance,output_shape = eucl_dist_output_shape))
eucl_dist_conv = Sequential(name="eucl_dist_conv")
eucl_dist_conv.add(Merge([conv_1,conv_2],mode=euclidean_distance,output_shape = eucl_dist_output_shape))

merged_model = Sequential(name="concat model")
merged_model.add(Merge([cos_dist_embedd, cos_dist_conv,cos_dist_lstm,
                        eucl_dist_embedd, eucl_dist_conv,eucl_dist_lstm
                        ], mode='concat'))
merged_model.add(BatchNormalization())
dense_layer(merged_model,num_dense,drop_dense)
dense_layer(merged_model,num_dense/2,drop_dense)
dense_layer(merged_model,num_dense/2,drop_dense)
dense_layer(merged_model,num_dense/4,drop_dense)
dense_layer(merged_model,num_dense/4,drop_dense)


merged_model.add(Dense(1))
merged_model.add(Activation('sigmoid'))


x_train_1,x_test_1,x_train_2,x_test_2,y_train,y_test = train_test_split(x1,x2,y,test_size=.1,random_state=4)
weight_val = np.ones(len(y_test))
class_weight = {0: 1.309028344, 1: 0.472001959}
weight_val *= 0.472001959
weight_val[y_test==0] = 1.309028344
early_stopping =EarlyStopping(monitor='val_loss', patience=5)
merged_model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])

checkpoint = ModelCheckpoint(str(STAMP+'.h5'), monitor='val_loss', save_best_only=True, verbose=2)
tbCallBack = keras.callbacks.TensorBoard(log_dir='./deepnet', histogram_freq=0, write_graph=True, write_images=True)
#print(merged_model.summary)
merged_model.fit(x=[x_train_1, x_train_2,x_train_1, x_train_2,x_train_1, x_train_2
                    ], y=y_train, batch_size=512, epochs=200,
                 verbose=1, class_weight=class_weight,
                 validation_data=[[x_test_1,x_test_2,x_test_1,x_test_2,x_test_1,x_test_2
                                   ],y_test,weight_val], shuffle=True, callbacks=[checkpoint,tbCallBack,
                                                                                  early_stopping])
prediction = merged_model.predict([x_test_1,x_test_2,x_test_1,x_test_2,x_test_1,x_test_2])
prediction = [1 if x >.5 else 0 for x in prediction]
print('f1_score:',(metrics.f1_score(prediction,y_test)))
