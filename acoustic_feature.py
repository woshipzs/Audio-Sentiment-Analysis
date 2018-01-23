#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 00:33:51 2017

@author: zeshan
"""
#generate audio features :
#   speech rate
#   # of pauses
#   decibels
#   pitch fundamentals - mean, covariance, standard deviation
import subprocess
import os
import csv
from pydub import AudioSegment

input_file = 'segmented data/'
output_file = 'features.txt'
abs_working_folder = 'audial signal sentiment analysis/'
praat = abs_working_folder + 'Praat.app/Contents/MacOS/Praat'
fzero = abs_working_folder + 'codes/fzero_driver'
acoustic = abs_working_folder + 'codes/single_syllable_count_hybrid'

in_path = abs_working_folder + input_file
out_path = abs_working_folder + output_file

silence_threshold = -50
min_dip_between_peaks = 2
min_pause_duration = 0.3
min_sounding_duration = 0.5
min_pitch = 70

def calc_features(in_file,out_file):
    audio = AudioSegment.from_file(in_file, format='wav')
    if len(audio) < 500:
        return ''
    
    bcommand = [praat, '--run', 'acoustic_features.praat', '0', str(silence_threshold), str(min_dip_between_peaks), 
                str(min_pause_duration), str(min_sounding_duration), str(min_pitch),
                'yes', in_file, out_file]
    execute = subprocess.Popen(bcommand)
    execute.communicate()
    if execute.poll() != 0:
        return ''
    
    f = open(out_path, 'r')
    line = f.readline()
    z = line.split(',')
    
    decibel = z[0]
    pitch_mean = z[1]
    pitch_sd = z[2]
    npause = z[3]
    speakingtot = z[4]
    speakingrate = z[5]
    duration = z[6]
    jitter = z[7]
    shimmer = z[8]
    mean_nhr = z[9]
    
    valid = True
    
    if jitter == '--undefined--' or shimmer == '--undefined--' or \
        mean_nhr == '--undefined--' or pitch_mean == '--undefined--' or \
        speakingrate == '0' or speakingrate == '--undefined--':
        valid = False
    
    if any(char.isdigit() for char in pitch_sd) and any(char.isdigit() for char in pitch_mean) and pitch_mean != '0':
        pitch_cov = 100*float(pitch_sd)/float(pitch_mean)
    else:
        pitch_cov = 'n/a'
        
    f.close()
    
    if valid:
        return [speakingrate,pitch_mean,pitch_cov,pitch_sd,npause,decibel]
    else:
        return ''
    

def calc_all_segments(in_path, out_path):
    csvfile = open(abs_working_folder+'acoustic_features.csv', 'wb')
    filewriter = csv.writer(csvfile, delimiter=',')
    filewriter.writerow(['Call ID','Time-ordered segment', 'Salesperson\'s speech rate', 'Mean Frequency',
        'Frequency CoV', 'Frequency SD', 'Pauses', 'Decibels', 'Customer\'s speech rate', 'Mean Frequency',
        'Frequency CoV', 'Frequency SD', 'Pauses', 'Decibels', 'Customer Sentiment', 'Meeting scheduled'])
    po_path = in_path + 'positive/'
    for folder in os.listdir(po_path):
        if folder != '.DS_Store':
            call_id = folder.split('.')[1]
            path = po_path + folder + '/'
            for audio in os.listdir(path):
                result = calc_features(path+audio, out_path)
                if result != '':
                    order = audio.split('.')[0]
                    who = audio.split('.')[1]
                    if who == 'salesperson':
                        filewriter.writerow([call_id, order] + result + ['n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', '1'])
                    elif who == 'customer':
                        filewriter.writerow([call_id, order, 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a'] + result + ['n/a', '1'])
                
            
    ne_path = in_path + 'negative/'
    for folder in os.listdir(ne_path):
        if folder != '.DS_Store':
            call_id = folder.split('.')[1]
            path = ne_path + folder + '/'
            for audio in os.listdir(path):
                result = calc_features(path+audio, out_path)
                if result != '':
                    order = audio.split('.')[0]
                    who = audio.split('.')[1]
                    if who == 'salesperson':
                        filewriter.writerow([call_id, order] + result + ['n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', '0'])
                    elif who == 'customer':
                        filewriter.writerow([call_id, order, 'n/a', 'n/a', 'n/a', 'n/a', 'n/a', 'n/a'] + result + ['n/a', '0'])
    csvfile.close()
        
        
calc_all_segments(in_path, out_path)