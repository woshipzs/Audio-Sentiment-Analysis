from pydub import AudioSegment
import numpy as np
import os
import csv

label_path = 'concatenated.csv'
seg_dir = 'segmented data/'
audio_dir = 'Inside Sales Call MP3s/'
trans_dir = 'transcripts/'
audio_format = 'mp3'
AudioSegment.converter = "/usr/local/Cellar/ffmpeg/3.4.1/bin/ffmpeg"

def convert_to_milliseconds(time):
    t = time.split(':')
    return int(int(t[0])*3600*1000+int(t[1])*60*1000+float(t[2])*1000)

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

def segment_one_audio(trans_path, audio_path, seg_loc, label):
    
    # trans_path: transcript path
    # audio_path: audio path
    # seg_loc: segmented audios folder path
    # label: the label for this audio
    
    basename = os.path.splitext(os.path.basename(trans_path))[0]
    p = seg_loc + label + '/' + basename
    if not os.path.isdir(p):
        os.makedirs(p)
    else:
        return
    audio = AudioSegment.from_file(audio_path, format=audio_format)
    trans = open(trans_path, 'r')
    lines = trans.readlines()
    n = len(lines)
    if n == 1:
        for line in lines:
            lines = line.split('\r\r')
            n = len(lines)
    a = 0
    
    has_start = False
    who = ''
    count = 0
    customer_count = 0
    salesperson_count = 0
    
    while(a < n):
        if lines[a] != '\r\n':
            words = lines[a].split(' ')
            for word in words:
                if '[' in word and ']' in word and any(char.isdigit() for char in word):
                    time = convert_to_milliseconds(word.replace('[', '').replace(']', '').replace('\r\n',''))
                    if not has_start:
                        start = time
                        has_start = True
                    else:
                        end = time
                        has_start = False
                       
                if 'customer' in word.lower():
                    who = 'customer'
                elif 'salesperson' in word.lower():
                    who = 'salesperson'
            if not has_start:
                if count >= 0 and who == 'customer':
                    seg = audio[start:end]
                    count = count + 1
                    customer_count = customer_count + 1
                    seg.export(p + '/' + str(count).zfill(3) + '.' + who + '.' + str(customer_count) + '.wav', format = 'wav')
                elif count > 0 and who == 'salesperson':
                    seg = audio[start:end]
                    count = count + 1
                    salesperson_count = salesperson_count + 1
                    seg.export(p + '/' + str(count).zfill(3) + '.' + who + '.' + str(salesperson_count) + '.wav', format = 'wav')
        
        a = a + 1
        
def segment_all_audios(trans_dir, audio_dir, seg_dir, label_path):
    
    # trans_dir: transcripts folder path
    # audio_dir: audios folder path
    # seg_dir: segmented audios folder path
    # label_path: one-hot label matrix for all audio files
    
    for trans in os.listdir(trans_dir):
        trans_path = os.path.join(trans_dir, trans)
        basename = os.path.splitext(trans)[0]
        audio_path = audio_dir + basename + '.' + audio_format 
        if not '.' in os.path.splitext(trans)[0] :
            filename = basename;
        else:
            filename = basename.split('.')[1]
        label_matrix = get_labels(label_path)
        label = find_label(filename, label_matrix)
        if label == 'N' or label == 'n':
            label = 'negative'
            segment_one_audio(trans_path, audio_path, seg_dir, label)
        elif label == 'Y' or label == 'y':
            label = 'positive'
            segment_one_audio(trans_path, audio_path, seg_dir, label)
        else:
            continue
                
segment_all_audios(trans_dir, audio_dir, seg_dir, label_path)
