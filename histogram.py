#!/usr/bin/env python2
# -*- coding: utf-8 -*-

#   this file generates histogram of audio files durations

import os
import numpy as np
import matplotlib.pyplot as plt
from pyAudioAnalysis import audioBasicIO

#   path to the audio files
path = '../ringtone removed labeled wav/'

his = []
count = 0
for f in os.listdir(path):
    if f != '.DS_Store':
        p = os.path.join(path,f)
        [Fs, x] = audioBasicIO.readAudioFile(p);
        duration = len(x)/float(Fs)
        if duration < 600 and count < 86:
            his.append(duration)
            count = count + 1
        
hist, bins = np.histogram(his, bins=50)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.xlabel('seconds')
plt.ylabel('counts')
plt.title('Histogram of Audio Record Durations')
plt.show()
        
