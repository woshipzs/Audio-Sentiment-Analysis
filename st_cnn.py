#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import scipy.io
import csv
import numpy as np
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from sklearn.model_selection import train_test_split
import keras.utils
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, Callback
from keras.utils import plot_model
from keras import initializers, layers



window_size = 200
label_path = 'RecordsGroundTruth.csv'
audio_dir = 'ringtone removed labeled wav/'


def short_term_features(in_path, window_size):
    [Fs, x] = audioBasicIO.readAudioFile(in_path)
    F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 1.0*Fs, 1.0*Fs);
    st = F.transpose()
    [r, c] = st.shape
    n = r/window_size
    re = r%window_size
    a = 0
    
    if n != 0:
        for i in range(n):
            if a == 0:
                a = 1
                f = np.array([st[0:window_size:1,:]])
            else:
                start = window_size*i
                g = st[start:start+window_size:1,:]
                h = np.array([g])
                f = np.vstack((f,h))
        
        remain = st[(n*window_size):(n*window_size+re):1,:]
        pudding = np.zeros(((window_size-re),c))
        m = np.vstack((remain,pudding))
        f = np.vstack((f,np.array([m])))
    else:
        remain = st[(n*window_size):(n*window_size+re):1,:]
        pudding = np.zeros(((window_size-re),c))
        m = np.vstack((remain,pudding))
        f = np.array([m])

    return f

def normalize(feature_matrix):
    return (feature_matrix-feature_matrix.min(0)) * 1.0 / (feature_matrix.ptp(0))
    

def one_hot_matrix(vector):
    one_hot = np.zeros((vector.size, max(max(vector))+1))
    one_hot[np.arange(vector.size), vector.flatten()] = 1
    return one_hot

def get_labels(path):
    csvfile = open(path, 'rb')
    reader = csv.reader(csvfile)
    
    #skip headers
    next(reader, None)
    
    rows = []
    for row in reader:
        rows.append(row)              
    return np.array(rows)

def find_label(filename, label_matrix):
    for row in label_matrix:
        if row[0] == filename:
            return row[1]
    
def data_generator(label_path,audio_dir,window_size):
    
    label_matrix = get_labels(label_path)
    a = 0
    
    for audio in os.listdir(audio_dir):
        if audio != '.DS_Store':
            filename = audio.split('.')[0]
            if '00T6000005HiT1f' in filename:
                filename = '00T6000005HiT1f'
            label = find_label(filename, label_matrix)
            if a == 0:
                a = 1
                features = short_term_features(audio_dir+audio,window_size)
                #features = normalize(features)
                n = len(features)
                if label == '1':
                    labels = np.ones(n)
                elif label == '0':
                    labels = np.zeros(n)
            else:
                f = short_term_features(audio_dir+audio,window_size)
                #f = normalize(f)
                features = np.vstack((features,f))
                n = len(f)
                if label == '1':
                    l = np.ones(n)
                    labels = np.hstack((labels,l))
                elif label == '0':
                    l = np.zeros(n)
                    labels = np.hstack((labels,l))
            
    features.dump('cnn_dataset_1_200.dat')
    labels.dump('cnn_labels_1_200.dat')
    print len(features),len(labels)
    return features, labels

class Length(layers.Layer):
    """
    Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss.
    Using this layer as model's output can directly predict labels by using `y_pred = np.argmax(model.predict(x), 1)`
    inputs: shape=[None, num_vectors, dim_vector]
    output: shape=[None, num_vectors]
    """
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))
    def compute_output_shape(self, input_shape):
        return input_shape[:-1]


class Mask(layers.Layer):
    """
    Mask a Tensor with shape=[None, num_capsule, dim_vector] either by the capsule with max length or by an additional
    input mask. Except the max-length capsule (or specified capsule), all vectors are masked to zeros. Then flatten the
    masked Tensor.
    For example:
        ```
        x = keras.layers.Input(shape=[8, 3, 2])  # batch_size=8, each sample contains 3 capsules with dim_vector=2
        y = keras.layers.Input(shape=[8, 3])  # True labels. 8 samples, 3 classes, one-hot coding.
        out = Mask()(x)  # out.shape=[8, 6]
        # or
        out2 = Mask()([x, y])  # out2.shape=[8,6]. Masked with true labels y. Of course y can also be manipulated.
        ```
    """
    def call(self, inputs, **kwargs):
        if type(inputs) is list:  # true label is provided with shape = [None, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, mask = inputs
        else:  # if no true label, mask by the max length of capsules. Mainly used for prediction
            # compute lengths of capsules
            x = K.sqrt(K.sum(K.square(inputs), -1))
            # generate the mask which is a one-hot code.
            # mask.shape=[None, n_classes]=[None, num_capsule]
            mask = K.one_hot(indices=K.argmax(x, 1), num_classes=x.get_shape().as_list()[1])
        # inputs.shape=[None, num_capsule, dim_capsule]
        # mask.shape=[None, num_capsule]
        # masked.shape=[None, num_capsule * dim_capsule]
        masked = K.batch_flatten(inputs * K.expand_dims(mask, -1))
        return masked
    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # no true label provided
            return tuple([None, input_shape[1] * input_shape[2]])


def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors


class CapsuleLayer(layers.Layer):
    """
    The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the
    neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
    from scalar to vector. So its input shape = [None, input_num_capsule, input_dim_capsule] and output shape = \
    [None, num_capsule, dim_capsule]. For Dense Layer, input_dim_capsule = dim_capsule = 1.
    :param num_capsule: number of capsules in this layer
    :param dim_capsule: dimension of the output vectors of the capsules in this layer
    :param num_routing: number of iterations for the routing algorithm
    """
    def __init__(self, num_capsule, dim_capsule, num_routing=3,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.num_routing = num_routing
        self.kernel_initializer = initializers.get(kernel_initializer)
    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]
        # Transform matrix
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W')
        self.built = True
    def call(self, inputs, training=None):
        # inputs.shape=[None, input_num_capsule, input_dim_capsule]
        # inputs_expand.shape=[None, 1, input_num_capsule, input_dim_capsule]
        inputs_expand = K.expand_dims(inputs, 1)
        # Replicate num_capsule dimension to prepare being multiplied by W
        # inputs_tiled.shape=[None, num_capsule, input_num_capsule, input_dim_capsule]
        inputs_tiled = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])
        # Compute `inputs * W` by scanning inputs_tiled on dimension 0.
        # x.shape=[num_capsule, input_num_capsule, input_dim_capsule]
        # W.shape=[num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]
        # Regard the first two dimensions as `batch` dimension,
        # then matmul: [input_dim_capsule] x [dim_capsule, input_dim_capsule]^T -> [dim_capsule].
        # inputs_hat.shape = [None, num_capsule, input_num_capsule, dim_capsule]
        inputs_hat = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]), elems=inputs_tiled)
        # Begin: Routing algorithm ---------------------------------------------------------------------#
        # In forward pass, `inputs_hat_stopped` = `inputs_hat`;
        # In backward, no gradient can flow from `inputs_hat_stopped` back to `inputs_hat`.
        inputs_hat_stopped = K.stop_gradient(inputs_hat)
        # The prior for coupling coefficient, initialized as zeros.
        # b.shape = [None, self.num_capsule, self.input_num_capsule].
        b = tf.zeros(shape=[K.shape(inputs_hat)[0], self.num_capsule, self.input_num_capsule])
        assert self.num_routing > 0, 'The num_routing should be > 0.'
        for i in range(self.num_routing):
            # c.shape=[batch_size, num_capsule, input_num_capsule]
            c = tf.nn.softmax(b, dim=1)
            # At last iteration, use `inputs_hat` to compute `outputs` in order to backpropagate gradient
            if i == self.num_routing - 1:
                # c.shape =  [batch_size, num_capsule, input_num_capsule]
                # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
                # The first two dimensions as `batch` dimension,
                # then matmal: [input_num_capsule] x [input_num_capsule, dim_capsule] -> [dim_capsule].
                # outputs.shape=[None, num_capsule, dim_capsule]
                outputs = squash(K.batch_dot(c, inputs_hat, [2, 2]))  # [None, 10, 16]
            else:  # Otherwise, use `inputs_hat_stopped` to update `b`. No gradients flow on this path.
                outputs = squash(K.batch_dot(c, inputs_hat_stopped, [2, 2]))
                # outputs.shape =  [None, num_capsule, dim_capsule]
                # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
                # The first two dimensions as `batch` dimension,
                # then matmal: [dim_capsule] x [input_num_capsule, dim_capsule]^T -> [input_num_capsule].
                # b.shape=[batch_size, num_capsule, input_num_capsule]
                b += K.batch_dot(outputs, inputs_hat_stopped, [2, 3])
        # End: Routing algorithm -----------------------------------------------------------------------#
        return outputs
    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])


def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    """
    Apply Conv2D `n_channels` times and concatenate all capsules
    :param inputs: 4D tensor, shape=[None, width, height, channels]
    :param dim_capsule: the dim of the output vector of capsule
    :param n_channels: the number of types of capsules
    :return: output tensor, shape=[None, num_capsule, dim_capsule]
    """
    output = layers.Conv2D(filters=dim_capsule * n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                           name='primarycap_conv2d')(inputs)
    outputs = layers.Reshape(target_shape=[-1, dim_capsule], name='primarycap_reshape')(output)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)

def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
    return K.mean(K.sum(L, 1))

def mk_dir(dir):
    try:
        os.makedirs(dir)
    except OSError:
        print('Can not make directory:', dir)


def defineExperimentPaths(basic_path, methodName, experimentID):
    experiment_name = methodName + '/' + experimentID
    MODEL_PATH = basic_path + experiment_name + '/model/'
    LOG_PATH = basic_path + experiment_name + '/logs/'
    CHECKPOINT_PATH = basic_path + experiment_name + '/checkpoints/'
    RESULT_PATH = basic_path + experiment_name + '/results/'
    mk_dir(MODEL_PATH)
    mk_dir(CHECKPOINT_PATH)
    mk_dir(RESULT_PATH)
    mk_dir(LOG_PATH)
    return [MODEL_PATH, CHECKPOINT_PATH, LOG_PATH, RESULT_PATH]



def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)


def storeResults(RESULT_PATH, trainingLossHistory,validataionLossHistory,testAcc, timeCost):
    results = open(os.path.join(RESULT_PATH, 'results.txt'), 'w')
    results.write("test acc:")
    results.write("\n")
    results.write(str(testAcc))
    results.write("\n")
    results.write("time cost:")
    results.write("\n")
    results.write(str(timeCost))
    results.write("\n")
    results.write("training loss:")
    results.write("\n")
    results.write(str(trainingLossHistory))
    results.write("\n")
    results.write("validation loss:")
    results.write("\n")
    results.write(str(validataionLossHistory))
    results.write("\n")
    results.close()
    return

def train_model(dataset, labels):
    
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.1, random_state = 227)
    
    n,h,w = dataset.shape
    input_shape = ( h, w, 1)
    X_train = X_train.reshape(X_train.shape[0], h, w, 1)
    X_test = X_test.reshape(X_test.shape[0], h, w, 1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    Y_train = keras.utils.to_categorical(y_train, 2)
    Y_test = keras.utils.to_categorical(y_test, 2)

    model = Sequential()
    model.add(Conv2D(32, (10, 1), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(5, 1)))

    model.add(Conv2D(64, (8, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(5, 1)))
    
    model.add(Conv2D(128, (5, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    #model.add(Dropout(0.25))
    
#    model.add(Conv2D(128, (5, 1)))
#    model.add(Activation('relu'))
#    model.add(MaxPooling2D(pool_size=(2, 1)))
    
#    model.add(Conv2D(32, (5, 1)))
#    model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 1)))
#    model.add(Dropout(0.25))

#    model.add(Conv2D(128, (2, 1)))
#    model.add(Activation('relu'))
#    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Dropout(0.25))

#    model.add(Flatten())
#    model.add(Dense(256))
#    model.add(Activation('relu'))
#    model.add(Dropout(0.5))
#    model.add(Dense(1))
#    model.add(Activation('sigmoid'))
    model.add(CapsuleLayer(num_capsule=2, dim_capsule=16, num_routing=1))
    model.add(Length())

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics={'capsnet': 'accuracy'})
    
    plot_model(model, to_file='model.png', show_shapes='True')
    #checkpointer = ModelCheckpoint(filepath='weights/smallest_loss_weights.hdf5', monitor='loss', verbose=2, save_best_only=True)
    history = model.fit(X_train, y_train, batch_size=50, nb_epoch=30, verbose=0, 
              validation_data=(X_test, y_test)) #callbacks=[checkpointer])
    #score = model.evaluate(X_test, y_test, verbose=1)
    #print score
    # summarize history for accuracy
    plt.subplot(1,2,1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # summarize history for loss
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    

#dataset, labels = data_generator(label_path,audio_dir,window_size)
dataset = np.load('cnn_dataset_0.05.dat')
labels = np.load('cnn_labels_0.05.dat')
train_model(dataset, labels)