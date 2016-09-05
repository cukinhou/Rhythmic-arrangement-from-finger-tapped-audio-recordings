import numpy as np
import scipy as sp
from sklearn.cluster import KMeans
import json
import essentia.standard as es
from essentia import array
from mido import Message, MidiFile, MidiTrack
from decimal import Decimal
import sys, getopt
def quantizedEnergy(slices, Q):

    rms = es.RMS()
    energy = [rms(item) for item in slices]
    energy = np.divide(energy, max(energy))

    # qEnergy = [Q*np.floor(item/Q + 0.5) for item in energy]
    qEnergy = [float(Decimal(item).quantize(Decimal(str(Q)), rounding='ROUND_DOWN')) for item in energy]
    qEnergy_out = []

    for i in range(0,len(slices)):
        if i == 0:
            qEnergy_out.append([0.0, qEnergy[i], qEnergy[i+1]])
        elif i == len(slices)-1:
            qEnergy_out.append([qEnergy[i-1], qEnergy[i], 0.0])

        else:
            qEnergy_out.append([qEnergy[i-1], qEnergy[i], qEnergy[i+1]])

    return qEnergy_out

def spectralCentroid (slice):

    w = es.Windowing(type='hann')
    fft = es.FFT()  # this gives us a complex FFT
    c2p = es.CartesianToPolar()  # and this turns it into a pair (magnitude, phase)
    centroid = es.Centroid()
    mean = es.Mean()
    specCentroid = []

    for frame in es.FrameGenerator(slice, frameSize=2048, hopSize=1024):
        mag, phase, = c2p(fft(w(frame)))
        specCentroid.append(centroid(mag))

    spectralCentroid = mean(specCentroid)
    return spectralCentroid

def extractFeatures(filename, dict):

    audio = es.MonoLoader(filename = filename)()
    filename = filename[filename.rfind('/')+1::]

    #Onset Detection
    audio = np.divide(audio, np.double(max(audio)))
    sFlux_onsets = es.SuperFluxExtractor(ratioThreshold = 9)
    onsets = sFlux_onsets(audio)

    end_times = list(onsets[1::])
    end_times.append(len(audio) / 44100.0)

    slices = es.Slicer(startTimes=array(onsets), endTimes=array(end_times))(audio)

    #Feature extraction
    data = []
    lowlevel_extractor = es.LowLevelSpectralExtractor()
    mean = es.Mean()
    idx = 0

    qEnergy = quantizedEnergy(slices, 0.1)


    duration = es.EffectiveDuration()

    for slice in slices:
        onset_lowlevel = lowlevel_extractor(slice)

        #Barkbands
        nBands = len(onset_lowlevel[0][0])
        barkbands = zip(*onset_lowlevel[0])
        bark_mean = []
        for i in range(0,nBands):
            bark_mean.append(mean(array(barkbands[i])))

        #MFCCs
        nMFCC = len(onset_lowlevel[5][0])
        mfcc = zip(*onset_lowlevel[5])
        mfcc_mean = []
        for i in range(0, nMFCC):
            mfcc_mean.append(mean(array(mfcc[i])))

        spectral_centroid = spectralCentroid(slice)

        # dict['data'].append([filename, idx, float(end_times[idx]), duration(slice), qEnergy[idx][1], qEnergy[idx][0], qEnergy[idx][2]])

        dict['data'].append([float(onsets[idx]), float(end_times[idx]), duration(slice),
                             qEnergy[idx][1], qEnergy[idx][0], qEnergy[idx][2],

                            bark_mean[0], bark_mean[1], bark_mean[2], bark_mean[3], bark_mean[4], bark_mean[5],
                           bark_mean[6], bark_mean[7], bark_mean[8], bark_mean[9], bark_mean[10], bark_mean[11],

                           bark_mean[12], bark_mean[13], bark_mean[14], bark_mean[15], bark_mean[16], bark_mean[17],
                           bark_mean[18], bark_mean[19], bark_mean[20], bark_mean[21], bark_mean[22], bark_mean[23],
                           bark_mean[24], bark_mean[25], bark_mean[26],

                           spectral_centroid,

                           mean(onset_lowlevel[1]), mean(onset_lowlevel[2]), mean(onset_lowlevel[3]), mean(onset_lowlevel[4]),

                           mfcc_mean[0], mfcc_mean[1], mfcc_mean[2], mfcc_mean[3], mfcc_mean[4], mfcc_mean[5],
                           mfcc_mean[6], mfcc_mean[7], mfcc_mean[8], mfcc_mean[9], mfcc_mean[10], mfcc_mean[11],
                           mfcc_mean[12],

                           mean(onset_lowlevel[9]), mean(onset_lowlevel[10]), mean(onset_lowlevel[11]),
                           mean(onset_lowlevel[12]), mean(onset_lowlevel[13]), mean(onset_lowlevel[14]),
                           mean(onset_lowlevel[15]), mean(onset_lowlevel[16]), mean(onset_lowlevel[17]),
                           mean(onset_lowlevel[18]), mean(onset_lowlevel[19]), mean(onset_lowlevel[20]),
                           mean(onset_lowlevel[21]), mean(onset_lowlevel[22]), mean(onset_lowlevel[23]),
                           mean(onset_lowlevel[24]), mean(onset_lowlevel[25]), mean(onset_lowlevel[26])

        ])

        idx +=1

    return dict

def toMIDI(filename, ch, notes, rms, onset_start_times, onset_end_times, nOnsets):

    print 'Transcribing to MIDI in ' + filename

    delta = 0
    with MidiFile() as outfile:
        track = MidiTrack()
        outfile.tracks.append(track)

        for i in range(nOnsets):
            stime = int((onset_start_times[i] - delta) * 1000)
            message = Message('note_on', note=int(notes[i]), velocity=int(rms[i] * 127), time=stime)
            message.channel = ch
            track.append(message)
            etime = int((onset_end_times[i] - delta) * 1000)
            off_message = Message('note_off', note=int(notes[i]), velocity=int(rms[i] * 127), time=etime)
            off_message.channel = ch
            track.append(off_message)
            delta = onset_end_times[i]

        outfile.print_tracks()
        outfile.save('/media/sf_VMshare/' + filename)

    print 'Transcription successfull!'
def main(argv):
    midifile = ''
    filename = ''
    dataset = ''
    try:

        opts, args = getopt.getopt(argv,"h:i:o:d:",["help", "ifile=","ofile=", "dataset="])
    except getopt.GetoptError:
        print 'Usage: dataset.py -i <inputfile.wav> -o <outputfile.mid> -d <dataset.json>'
        sys.exit(2)

    for opt, arg in opts:

        if opt == '-h':
            print 'MIDItranscription.py -i <inputfilename.wav> -o <outputfilename.mid> -d <dataset.json>'
            sys.exit()
        elif opt in ('-i', "--ifile"):
            filename = arg
        elif opt in ('-o', "--ofile"):
            midifile = arg

        elif opt in ('-d', "--dataset"):
            dataset = arg

        else:
          print 'ERROR: Input or Output files are not .wav or .mid respectevely'
          sys.exit(2)



    dict = {
        'description': 'Finger-tapped_dataset',
        'relation': 'Feature_onsets',
        'attributes': [

            ('Onset_start', 'REAL'), ('Onset_end', 'REAL'), ('Effective_duration', 'REAL'),

            ('qRMS', 'REAL'), ('qPrevRMS', 'REAL'), ('qNextRMS', 'REAL'),

            ('Bark_band1.mean', 'REAL'), ('Bark_band2.mean', 'REAL'), ('Bark_band3.mean', 'REAL'),
            ('Bark_band4.mean', 'REAL'), ('Bark_band5.mean', 'REAL'), ('Bark_band6.mean', 'REAL'),
            ('Bark_band7.mean', 'REAL'), ('Bark_band8.mean', 'REAL'), ('Bark_band9.mean', 'REAL'),
            ('Bark_band10.mean', 'REAL'), ('Bark_band11.mean', 'REAL'), ('Bark_band12.mean', 'REAL'),
            ('Bark_band13.mean', 'REAL'), ('Bark_band14.mean', 'REAL'), ('Bark_band115.mean', 'REAL'),
            ('Bark_band16.mean', 'REAL'), ('Bark_band17.mean', 'REAL'), ('Bark_band18.mean', 'REAL'),
            ('Bark_band19.mean', 'REAL'), ('Bark_band20.mean', 'REAL'), ('Bark_band21.mean', 'REAL'),
            ('Bark_band22.mean', 'REAL'), ('Bark_band23.mean', 'REAL'), ('Bark_band24.mean', 'REAL'),
            ('Bark_band25.mean', 'REAL'), ('Bark_band26.mean', 'REAL'), ('Bark_band27.mean', 'REAL'),

            ('Spectral_centroid', 'REAL'),

            ('Bark_band_Kurtosis.mean', 'REAL'), ('Bark_band_Skewness.mean', 'REAL'), ('Bark_band_Spread.mean', 'REAL'),
            ('HFC.mean', 'REAL'),

            ('MFCC_0.mean', 'REAL'), ('MFCC_1.mean', 'REAL'), ('MFCC_2.mean', 'REAL'), ('MFCC_3.mean', 'REAL'),
            ('MFCC_4.mean', 'REAL'), ('MFCC_5.mean', 'REAL'),
            ('MFCC_6.mean', 'REAL'), ('MFCC_7.mean', 'REAL'), ('MFCC_8.mean', 'REAL'), ('MFCC_9.mean', 'REAL'),
            ('MFCC_10.mean', 'REAL'), ('MFCC_11.mean', 'REAL'),
            ('MFCC_12.mean', 'REAL'),

            ('Silence_Rate_20dB.mean', 'REAL'), ('Silence_Rate_30dB.mean', 'REAL'), ('Silence_Rate_60dB.mean', 'REAL'),
            ('Spectral_Complex.mean', 'REAL'), ('Spectral_Crest.mean', 'REAL'), ('Spectral_Decrease.mean', 'REAL'),
            ('Spectral_Energy.mean', 'REAL'), ('Spectral_Energy_Low.mean', 'REAL'), ('Spectral_Energy_midLow.mean', 'REAL'),
            ('Spectral_Energy_midHigh.mean', 'REAL'), ('Spectral_Energy_high.mean', 'REAL'),
            ('Spectral_Flatness.mean', 'REAL'),
            ('Spectral_Flux.mean', 'REAL'), ('Spectral_RMS.mean', 'REAL'), ('Spectral_Rolloff.mean', 'REAL'),
            ('Spectral_Strong_Peak.mean', 'REAL'), ('ZeroCrossingRate.mean', 'REAL'), ('inharmonicity.mean', 'REAL')
        ],
        'data': []
    }

   
    data = json.load(open(dataset, 'rb'))

    x = data['data']

    attributes = np.array(data['attributes'])
    attributes = [items[0] for items in attributes[4:]]

    data_set = [items[4:] for items in x]

    est = KMeans(n_clusters=3)
    d = []

    for items in data_set:
        onset = [items[attributes.index('qRMS')],items[attributes.index('qPrevRMS')],items[attributes.index('qNextRMS') ],
                 items[attributes.index('Effective_duration')],items[attributes.index('Spectral_centroid')],
                 items[attributes.index('Spectral_Energy.mean')],       items[attributes.index('Spectral_Flatness.mean') ] ]
        d.append(onset)


    est.fit(d)

    y = extractFeatures(filename, dict)

    input = [items[2:] for items in y['data']]

    c = []
    for items in input:
        onset = [items[attributes.index('qRMS')],items[attributes.index('qPrevRMS')],items[attributes.index('qNextRMS') ],
                 items[attributes.index('Effective_duration')],items[attributes.index('Spectral_centroid')],
                 items[attributes.index('Spectral_Energy.mean')],                       items[attributes.index('Spectral_Flatness.mean') ] ]
        c.append(onset)

    notes = est.predict(c)

    notes = [36 if note==0 else note for note in notes]
    notes = [37 if note==1 else note for note in notes]
    notes = [38 if note==2 else note for note in notes]
    rms = [items[3] for items in y['data']]
	
    rms = np.divide(rms,max(rms))

    start_times = [items[0] for items in y['data']]

    end_times = [items[1] for items in y['data']]

    toMIDI(midifile, 10, notes, rms, start_times, end_times, len(y['data']))
 

if __name__ == "__main__":
    main(sys.argv[1:])
