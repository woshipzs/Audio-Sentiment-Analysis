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
    
    # this function calculates short-term features of one audio file
    #
    # in_path: input audio file address
    # window_size: length of the feature matrix
    
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
    
    # this function generates short-term feature matrixes
    #
    # label_path: ground-truth file address
    # audio_dir: directory that stores all audio files
    # window_size: length of the feature matrix
    
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

def train_model(dataset, labels):
    
    # this function trains a deep neural network by feeding feature matrixes
    #
    # dataset: training data address
    # labels: training label address
    
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
    

dataset, labels = data_generator(label_path,audio_dir,window_size)
dataset = np.load('cnn_dataset_0.05.dat')
labels = np.load('cnn_labels_0.05.dat')
train_model(dataset, labels)
