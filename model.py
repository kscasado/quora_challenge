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
from keras.layers import Concatenate, Input, concatenate
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

embedding_matrix = pickle.load(open('data/embedd_matrix.p','rb'))
x1 = pickle.load(open('data/q1_train.p','rb'))
x2 = pickle.load(open('data/q2_train.p','rb'))
y = pd.read_csv('data/train.csv')['is_duplicate']
word_index = embedding_matrix[0]-1
max_features = 200000
filter_length = 5
nb_filter = 64
pool_length = 4
EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 30
model = Sequential()
print('Build model...')
embedding_layer = Embedding(embedding_matrix.shape[0],
        300,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False)
model1 = Sequential()
model1.add(embedding_layer)

model1.add(TimeDistributed(Dense(300, activation='relu')))
model1.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(300,)))

model2 = Sequential()
model2.add(embedding_layer)

model2.add(TimeDistributed(Dense(300, activation='relu')))
model2.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(300,)))

model3 = Sequential()
model3.add(embedding_layer)
model3.add(Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1))
model3.add(Dropout(0.2))

model3.add(Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1))

model3.add(GlobalMaxPooling1D())
model3.add(Dropout(0.2))

model3.add(Dense(300))
model3.add(Dropout(0.2))
model3.add(BatchNormalization())

model4 = Sequential()
model4.add(embedding_layer)
model4.add(Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1))
model4.add(Dropout(0.2))

model4.add(Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1))

model4.add(GlobalMaxPooling1D())
model4.add(Dropout(0.2))

model4.add(Dense(300))
model4.add(Dropout(0.2))
model4.add(BatchNormalization())
model5 = Sequential()
model5.add(embedding_layer)
model5.add(Bidirectional(LSTM(300, dropout=0.2, recurrent_dropout=0.2,return_sequences=True)))
model5.add(AttentionWithContext())
model6 = Sequential()
model6.add(embedding_layer)
model6.add(Bidirectional(LSTM(300, dropout=0.2, recurrent_dropout=0.2,return_sequences=True)))
model6.add(AttentionWithContext())
merged_model = Sequential()
merged_model.add(Merge([model1, model2, model3, model4, model5, model6],mode='concat'))
merged_model.add(BatchNormalization())

merged_model.add(Dense(300))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())

merged_model.add(Dense(300))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())

merged_model.add(Dense(300))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())

merged_model.add(Dense(300))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())

merged_model.add(Dense(300))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())

merged_model.add(Dense(1))
merged_model.add(Activation('sigmoid'))

merged_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph_2_lstm_with_f1', histogram_freq=0, write_graph=True, write_images=True)
early_stopping =EarlyStopping(monitor='val_loss', patience=4)

checkpoint = ModelCheckpoint('weights.h5', monitor='val_acc', save_best_only=True, verbose=2)
#x_train_1,x_test_1,x_train_2,x_test_2,y_train,y_test = train_test_split(x1,x2,y,test_size=.1,random_state=4)
#merged_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
merged_model.fit([x1, x2, x1, x2, x1, x2], y=y, batch_size=384, nb_epoch=200,
                 verbose=1, validation_split=0.1, shuffle=True, callbacks=[checkpoint])
#merged_model.fit([x1,x2,x1,x2,x1,x2],y=y,
         # batch_size=256,epochs=100,validation_split=.1,shuffle=True,callbacks=[tbCallBack,early_stopping])
#tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph_2_lstm_with_f1', histogram_freq=0, write_graph=True, write_images=True)
#early_stopping =EarlyStopping(monitor='val_loss', patience=4)
#merged_model.fit([x1, x2, x1, x2, x1, x2], y=y, batch_size=384, epochs=200,
#                 verbose=1, validation_split=0.1, shuffle=True, callbacks=[checkpoint,tbCallBack])


#prediction = merged_model.predict([x_train_1,x_train_2,x_train_1,x_train_2,x_train_1,x_train_2])
print('f1_score',metrics.f1_score(prediction.argmax(axis=1),y_test[:,1]))
