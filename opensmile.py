#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import subprocess
import os
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dropout, Dense

gt_csv = '../RecordsGroundTruth.csv'
config = '../../../opensmile-2.3.0/config/IS09_emotion.conf'
wavs = '../ringtone removed labeled wav'
smilextract = '../../../opensmile-2.3.0/SMILExtract'
out_dir = 'opensmile'
in_dir = '../diarization'
epochs = 50
batch_size = 50

def get_labels(path):
#   this function generates one-hot ground-truth matrix
#
#   path: ground-truth file address

    csvfile = open(path, 'rb')
    reader = csv.reader(csvfile)
    
    #skip headers
    next(reader, None)
    
    rows = []
    for row in reader:
        rows.append(row)              
    return np.array(rows)

def generate_opensmile_features(in_dir, out_dir, gt_csv):
#   this function generates features of audio files using opensmile library
#
#   in_dir: directory that stores all audio files
#   out_dir: directory that stores the result file
#   gt_csv: ground-truth csv file address

    negcsv = 'negfeatures.csv'
    poscsv = 'posfeatures.csv'
    
    label_matrix = get_labels(gt_csv)
    
    for audio in os.listdir(in_dir):
        if audio != '.DS_Store' and audio != 'data':
            basename = audio.split('.')[0]
            if '00T6000005HiT1f' in basename:
                basename = '00T6000005HiT1f'
            n,c = np.where(label_matrix == basename)
            
            if label_matrix[n[0]][1] == '0':
                csvout = negcsv
            else:
                csvout = poscsv
            
            call_path = os.path.join(in_dir, audio)
        
            opensmile = subprocess.Popen([smilextract, 
                                      '-C', config, 
                                      '-instname', audio, 
                                      '-I', call_path, 
                                      '-csvoutput', os.path.join(out_dir, csvout), 
                                      '-timestampcsv', '0'],
                                        stdout=subprocess.PIPE)
            opensmile.communicate()
        
        
    neg_features = pd.read_csv(os.path.join(out_dir, negcsv), sep=';')
    pos_features = pd.read_csv(os.path.join(out_dir, poscsv), sep=';')
    neg_features['label'] = np.zeros((neg_features.shape[0], 1))
    pos_features['label'] = np.ones((pos_features.shape[0], 1))
    
    combined_features = neg_features.append(pos_features)
    combined_features.to_csv(os.path.join(out_dir, 'opensmile_fc.csv'), index=False, header=None)
    
def train_model(fc_loc):
#   three-layer neural network for trainning features from opensmile
#
#   fc_loc: directory that stores ground-truth file

    opensmile_fc = pd.read_csv(os.path.join(fc_loc, 'opensmile_fc.csv'), header=None)
    #ids = opensmile_fc.iloc[:, 0].values
    labels = opensmile_fc.iloc[:, -1].values
    data = opensmile_fc.iloc[:, 1:-1].values
    
    train_data, validation_data, train_labels, validation_labels = train_test_split(data, labels, test_size=0.1, random_state=42)


    model = Sequential()
    model.add(Dense(128, input_shape=train_data.shape[1:], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))

opensmile_fc = generate_opensmile_features(in_dir,out_dir,gt_csv)
train_model(out_dir)
