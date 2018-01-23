form Acoustic Parameters
    real Pitch_time_step_(s) 0.0
	real Silence_threshold_(dB) -25
   	real Minimum_dip_between_peaks_(dB) 2
   	real Minimum_pause_duration_(s) 0.3
   	real Minimum_sounding_duration_(s) 0.1
   	real Minimum_pitch_(Hz) 50
   	boolean Keep_Soundfiles_and_Textgrids yes
    sentence first_text ../ringtone removed labeled wav/00T6000005Gxkbr.wav
    sentence second_text praat_features.txt
endform

pitchts = 'Pitch_time_step'
silencedb = 'silence_threshold'
mindip = 'minimum_dip_between_peaks'
showtext = 'keep_Soundfiles_and_Textgrids'
minpause = 'minimum_pause_duration'
minsounding = 'minimum_sounding_duration'
minpitch = 'Minimum_pitch'


sound = Read from file: first_text$
soundname$ = selected$("Sound")
soundid = selected("Sound")
intens = Get intensity (dB)

pitch = To Pitch: pitchts, 75, 600
Down to PitchTier
mn = Get mean (curve): 0, 0
st = Get standard deviation (points): 0, 0

pointprocess = Down to PointProcess
selectObject: sound,pitch,pointprocess
report$ = Voice report: 0, 0, 75, 500, 1.3, 1.6, 0.03, 0.45
jitter_loc = extractNumber (report$, "Jitter (local): ") * 10e13
shimmer_loc = extractNumber (report$, "Shimmer (local): ") *100
mean_nhr = extractNumber (report$, "Mean noise-to-harmonics ratio: ")

# use object ID
   selectObject: "Sound " + soundname$

   originaldur = Get total duration
   # allow non-zero starting time
   bt = Get starting time

   # Use intensity to get threshold
   To Intensity... minpitch 0 yes
   intid = selected("Intensity")
   start = Get time from frame number... 1
   nframes = Get number of frames
   end = Get time from frame number... 'nframes'

   # estimate noise floor
   minint = Get minimum... 0 0 Parabolic
   # estimate noise max
   maxint = Get maximum... 0 0 Parabolic
   #get .99 quantile to get maximum (without influence of non-speech sound bursts)
   max99int = Get quantile... 0 0 0.99

   # estimate Intensity threshold
   threshold = max99int + silencedb
   threshold2 = maxint - max99int
   threshold3 = silencedb - threshold2
   if threshold < minint
       threshold = minint
   endif

  # get pauses (silences) and speakingtime
   To TextGrid (silences)... threshold3 minpause minsounding silent sounding
   textgridid = selected("TextGrid")
   silencetierid = Extract tier... 1
   silencetableid = Down to TableOfReal... sounding
   nsounding = Get number of rows
   npauses = 'nsounding'
   speakingtot = 0
   for ipause from 1 to npauses
      beginsound = Get value... 'ipause' 1
      endsound = Get value... 'ipause' 2
      speakingdur = 'endsound' - 'beginsound'
      speakingtot = 'speakingdur' + 'speakingtot'
   endfor

   select 'intid'
   Down to Matrix
   matid = selected("Matrix")
   # Convert intensity to sound
   To Sound (slice)... 1
   sndintid = selected("Sound")

   # use total duration, not end time, to find out duration of intdur
   # in order to allow nonzero starting times.
   intdur = Get total duration
   intmax = Get maximum... 0 0 Parabolic

   # estimate peak positions (all peaks)
   To PointProcess (extrema)... Left yes no Sinc70
   ppid = selected("PointProcess")

   numpeaks = Get number of points

   # fill array with time points
   for i from 1 to numpeaks
       t'i' = Get time from index... 'i'
   endfor 


   # fill array with intensity values
   select 'sndintid'
   peakcount = 0
   for i from 1 to numpeaks
       value = Get value at time... t'i' Cubic
       if value > threshold
             peakcount += 1
             int'peakcount' = value
             timepeaks'peakcount' = t'i'
       endif
   endfor


   # fill array with valid peaks: only intensity values if preceding 
   # dip in intensity is greater than mindip
   select 'intid'
   validpeakcount = 0
   currenttime = timepeaks1
   currentint = int1

   for p to peakcount-1
      following = p + 1
      followingtime = timepeaks'following'
      dip = Get minimum... 'currenttime' 'followingtime' None
      diffint = abs(currentint - dip)

      if diffint > mindip
         validpeakcount += 1
         validtime'validpeakcount' = timepeaks'p'
      endif
         currenttime = timepeaks'following'
         currentint = Get value at time... timepeaks'following' Cubic
   endfor


   # Look for only voiced parts
   select 'soundid' 
   To Pitch (ac)... 0.02 30 4 no 0.03 0.25 0.01 0.35 0.25 450
   # keep track of id of Pitch
   pitchid = selected("Pitch")

   voicedcount = 0
   for i from 1 to validpeakcount
      querytime = validtime'i'

      select 'textgridid'
      whichinterval = Get interval at time... 1 'querytime'
      whichlabel$ = Get label of interval... 1 'whichinterval'

      select 'pitchid'
      value = Get value at time... 'querytime' Hertz Linear

      if value <> undefined
         if whichlabel$ = "sounding"
             voicedcount = voicedcount + 1
             voicedpeak'voicedcount' = validtime'i'
         endif
      endif
   endfor

   
   # calculate time correction due to shift in time for Sound object versus
   # intensity object
   timecorrection = originaldur/intdur

   # Insert voiced peaks in TextGrid
   if showtext > 0
      select 'textgridid'
      Insert point tier... 1 syllables
      
      for i from 1 to voicedcount
          position = voicedpeak'i' * timecorrection
          Insert point... 1 position 'i'
      endfor
   endif

   # clean up before next sound file is opened
    select 'intid'
    plus 'matid'
    plus 'sndintid'
    plus 'ppid'
    plus 'pitchid'
    plus 'silencetierid'
    plus 'silencetableid'

    Remove
    if showtext < 1
       select 'soundid'
       plus 'textgridid'
       Remove
    endif

# summarize results in Info window
   speakingrate = 'voicedcount'/'originaldur'
   articulationrate = 'voicedcount'/'speakingtot'
   npause = 'npauses'-1
   asd = 'speakingtot'/'voicedcount'
   
#writeFileLine: second_text$, intens, "," ,mn, "," , st , "," , voicedcount , "," , npause , "," , originaldur , "," , speakingtot , "," , speakingrate , "," , articulationrate , "," , asd
writeFileLine: second_text$, intens, "," ,mn, "," , st , "," , npause , ","  , speakingtot , "," , speakingrate, "," , originaldur, ",", jitter_loc, ",", shimmer_loc, ",", mean_nhr

