#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 14:59:00 2017

@author: zeshan
"""

from pyAudioAnalysis import audioSegmentation as aS
from pydub import AudioSegment
import os

working_directory = '/Users/zeshan/Desktop/audial signal sentiment analysis/'
ringtone_removed_folder = 'ringtone removed labeled wav/'
diarization_folder = 'diarization/'

def split_call_into_speakers(in_loc, out_loc):
    
    for audio in os.listdir(in_loc):
        if audio != '.DS_Store':
            p = os.path.join(in_loc, audio)
            no_rings_audio = AudioSegment.from_file(p, format='wav')
            basename = os.path.splitext(os.path.basename(audio))[0] 
            # split on speakers now setting num speakers to 2
            diarized = aS.speakerDiarization(p, 2, mtSize=0.5, mtStep=0.1)
            # determine which label was given to customer and salesperson
            cust = diarized[0]
            # output the segments
            segs, flags = aS.flags2segs(diarized, 0.1)  #mtstep from above
            for seg in range(segs.shape[0]):
                # skip segments shorter than 1s (usually 'um' or something)
                if segs[seg, 1] - segs[seg, 0] < 1:
                    continue
                out_seg = no_rings_audio[segs[seg, 0]*1000:segs[seg, 1]*1000]
                if flags[seg] == cust:
                    out_seg.export(out_loc + basename + '.' + str(seg) + '_cust.wav', format='wav')
                else:
                    out_seg.export(out_loc + basename + '.' + str(seg) + '_sales.wav', format='wav')
            
split_call_into_speakers(working_directory+ringtone_removed_folder, working_directory+diarization_folder)
