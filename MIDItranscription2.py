# from argparse import ArgumentParser as ap
import sys, getopt
import essentia.standard as es
from mido import Message, MidiFile, MidiTrack
from essentia import array
from numpy import random, max

def computeDescriptors(filepath):

    sampleRate = 44100
    frameSize = 2048
    hopSize = 512

    audio = es.MonoLoader(filename=filepath)()
    filename = filepath[filepath.rfind('/') + 1:]

    print 'File: ' + filename

    w = es.Windowing(type='hann')
    fft = es.FFT()  # this gives us a complex FFT
    c2p = es.CartesianToPolar()  # and this turns it into a pair (magnitude, phase)
    odFlux = es.OnsetDetection(method='flux')

    flux = []

    print 'Onset detection...'

    for frame in es.FrameGenerator(audio, frameSize, hopSize):
        mag, phase, = c2p(fft(w(frame)))
        flux.append(odFlux(mag, phase))
    onset_times = es.Onsets()([array(flux)], [1])

    onset_start_times = list(onset_times)
    onset_end_times = onset_start_times[1::] + [len(audio) / 44100.0]

    print 'Slicing audio onsets...'

    slices = es.Slicer(startTimes = array(onset_start_times), endTimes = array(onset_end_times), timeUnits = 'seconds')(audio)
    number_onsets = len(slices)

    print 'Computing descriptors...'

    rms = es.RMS()
    centroid = es.Centroid()
    duration = es.Duration()
    bark = es.BarkBands(numberBands=4)
    tempo = es.BpmHistogramDescriptors()(audio)[0]

    mean = es.Mean()
    var = es.Variance()

    descriptors = {'filename': filename,
                   'Tempo':tempo, 'nOnsets': number_onsets,
                   'onset_start_times': onset_start_times,
                   'onset_end_times': onset_end_times,
                   'duration': [], 'RMS.mean': [], 'RMS.var': [],
                   'spectral_centroid.mean': [], 'spectral_centroid.var': [],
                   'bark_band1.mean': [], 'bark_band2.mean': [], 'bark_band3.mean': [],
                   'bark_band4.mean': [], 'bark_band1.var': [], 'bark_band2.var': [],
                   'bark_band3.var': [], 'bark_band4.var': []
                   }

    n = 0
    for slice in slices:
        bark_bands = []
        spectral_centroid = []
        RMS = []

        for frame in es.FrameGenerator(slice, frameSize, hopSize):

            mag, phase = c2p(fft(w(frame)))
            RMS.append(rms(frame))
            spectral_centroid.append(centroid(mag))
            bark_bands.append(bark(mag))

        descriptors['duration'].append(duration(slice))
        descriptors['RMS.mean'].append(mean(RMS))
        descriptors['RMS.var'].append(var(RMS))

        descriptors['spectral_centroid.mean'].append(mean(spectral_centroid))
        descriptors['spectral_centroid.var'].append(var(spectral_centroid))

        descriptors['bark_band1.mean'].append(mean(array([item[0] for item in bark_bands])))
        descriptors['bark_band2.mean'].append(mean(array([item[1] for item in bark_bands])))
        descriptors['bark_band3.mean'].append(mean(array([item[2] for item in bark_bands])))
        descriptors['bark_band4.mean'].append(mean(array([item[3] for item in bark_bands])))

        descriptors['bark_band1.var'].append(var(array([item[0] for item in bark_bands])))
        descriptors['bark_band2.var'].append(var(array([item[1] for item in bark_bands])))
        descriptors['bark_band3.var'].append(var(array([item[2] for item in bark_bands])))
        descriptors['bark_band4.var'].append(var(array([item[3] for item in bark_bands])))
        n += 1

    return descriptors


def normalize(descriptors):

    print 'Normalizing...'

    for key in descriptors:
        if key == 'duration' or key =='RMS.mean' or key == 'RMS.var' or key == 'spectral_centroid.mean' or key == 'spectral_centroid.var' or key == 'bark_band1.mean' or key == 'bark_band2.mean' or key == 'bark_band3.mean':
            descriptors[key] = [elements/max(descriptors[key]) for elements in descriptors[key]]

    return descriptors



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
    audiofile = ''

    midifile = ''

    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print 'Usage: MIDItranscription.py -i <inputfilename.wav> -o <outputfilename.mid>'
        sys.exit(2)
    for opt, arg in opts:
      if opt == '-h':
         print 'MIDItranscription.py -i <inputfilename.wav> -o <outputfilename.mid>'
         sys.exit()
      elif opt in ("-i", "--ifile") and arg.lower().endswith('.wav'):
         audiofile = arg
      elif opt in ("-o", "--ofile") and arg.lower().endswith('.mid'):
         midifile = arg

      else:
          print 'ERROR: Input or Output files are not .wav or .mid respectevely'
          sys.exit(2)

    descriptors = computeDescriptors(audiofile)

    descriptors = normalize(descriptors)
    notes = random.randint(50, 60, descriptors['nOnsets'])
    ch = 10
    toMIDI(midifile, ch, notes, descriptors['RMS.mean'],
           descriptors['onset_start_times'], descriptors['onset_end_times'], descriptors['nOnsets'])

    print 'Input file is "', audiofile
    print 'Output file is "', midifile

if __name__ == "__main__":
    main(sys.argv[1:])