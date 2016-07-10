import csv
import essentia.standard as es
from essentia import array
from os import listdir
import arff
import json
import sys


class dataset:

    def __init__(self, path):

        self.path = path
        self.data = {}

        list_files = listdir(path)
        list_files = [file for file in list_files if (file.endswith('.WAV') or file.endswith('.wav'))]

        for file in list_files:
            self.data[file] = {}

    def saveCSV(self, path):
        w = csv.writer(open(path, "w"))
        for key, val in self.data.items():
            w.writerow([key, val])

    def saveARFF(self, path):
        dataSet = arff.dumps(self.data)
        f = open(path+'.arff', 'wb')
        f.write(dataSet)


    def saveJSON(self, path):
        f = open(path+'.json', 'wb')
        json.dump(self.data, f)
    def normalize(self, vector):
        vector = [items/max(items) for items in vector]
        return vector
    def computeEndTimes(self, ):
        return ''
    def compute_statistics(self):
        return
    def computeDescriptors(self):
        sampleRate = 44100
        frameSize = 2048
        hopSize = 512
        numBands = 24

        w = es.Windowing(type='hann')
        fft = es.FFT()  # this gives us a complex FFT
        c2p = es.CartesianToPolar()  # and this turns it into a pair (magnitude, phase)
        odFlux = es.OnsetDetection(method='flux')

        rms = es.RMS()
        centroid = es.Centroid()
        duration = es.EffectiveDuration()
        bark = es.BarkBands(numberBands=numBands)
        bpm = es.BpmHistogramDescriptors()
        onsets = es.Onsets()
        mean = es.Mean()
        var = es.Variance()
        superflux = es.SuperFluxExtractor()

        for file in self.data.keys():
            print 'File name: ' + self.path + str(file)
            loader = es.MonoLoader(filename=self.path + file)
            audio = loader()

            flux = []

            # for frame in es.FrameGenerator(audio, frameSize, hopSize):
            #     mag, phase, = c2p(fft(w(frame)))
            #     flux.append(odFlux(mag, phase))
            #
            # onset_times = onsets([array(flux)], [1])

            onset_times = superflux(audio)
            end_times = list(onset_times)

            del end_times[0]
            end_times.append(len(audio) / 44100.0)

            slicer = es.Slicer(startTimes=array(onset_times), endTimes=array(end_times))
            slices = slicer(audio)

            self.data[file]['Tempo'] = bpm(audio)[0]

            nOnset = 0

            print 'Computing descriptors...'
            self.data[file]['slices'] = {}

            # Create cell for each Slice in the file
            for idx in range(0, len(slices)):
                self.data[file]['slices'][str(idx)] = {}

            # Index for accessing each slice
            idx = 0
            for slice in slices:

                # Create bark_bands cell for this slice
                for band in range(0, numBands):
                    self.data[file]['slices'][str(idx)]['Bark_band.'+str(band)+'.mean'] = []

                self.data[file]['slices'][str(idx)]['Spectral_centroid.mean'] = []
                self.data[file]['slices'][str(idx)]['RMS.mean'] = []



                for frame in es.FrameGenerator(slice, frameSize, hopSize):

                    mag, phase = c2p(fft(w(frame)))
                    self.data[file]['slices'][str(idx)]['RMS.mean'].append(rms(frame))
                    self.data[file]['slices'][str(idx)]['Spectral_centroid.mean'].append(centroid(mag))
                    frame_bark_bands = bark(mag)

                    for band in range(0, numBands):
                        self.data[file]['slices'][str(idx)]['Bark_band.' + str(band)+'.mean'].append(frame_bark_bands[band])

                self.data[file]['slices'][str(idx)]['Duration'] =  duration(slice)


                # COMPUTE STATISTICS
                for band in range(0, numBands):
                    barkband = array(self.data[file]['slices'][str(idx)]['Bark_band.' + str(band)+'.mean'])
                    self.data[file]['slices'][str(idx)]['Bark_band.' + str(band)+'.mean'] = mean(barkband)
                    self.data[file]['slices'][str(idx)]['Bark_band.' + str(band)+'.var'] = var(barkband)

                RMS = self.data[file]['slices'][str(idx)]['RMS.mean']
                self.data[file]['slices'][str(idx)]['RMS.mean'] = mean(RMS)
                self.data[file]['slices'][str(idx)]['RMS.var'] = var(RMS)

                cent = self.data[file]['slices'][str(idx)]['Spectral_centroid.mean']
                self.data[file]['slices'][str(idx)]['Spectral_centroid.mean'] = mean(cent)
                self.data[file]['slices'][str(idx)]['Spectral_centroid.var'] = var(cent)
                idx +=1



def main(argv):
    path = ''
    name = ''


    # try:
    #     opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    # except getopt.GetoptError:
    #     print 'Usage: dataset.py -p <path> -n <dataset_name>'
    #     sys.exit(2)
    # for opt, arg in opts:
    #   if opt == '-h':
    #      print 'MIDItranscription.py -i <inputfilename.wav> -o <outputfilename.mid>'
    #      sys.exit()
    #   elif opt in ("-i", "--path"):
    #
    #      #  !!!!!!!!!!!!!!!!!!!1
    #      path = '/media/sf_VMshare/Tapping/Audios/'
    #
    #   elif opt in ("-o", "--name"):
    #      name = arg
    #   else:
    #       print 'ERROR: Input or Output files are not .wav or .mid respectevely'
    #       sys.exit(2)
    path = '/media/sf_VMshare/Tapping/Audios/'
    set = dataset(path)

    set.computeDescriptors()
    print set.data
    set.saveJSON('pruebaV3')
    # set.saveARFF('arffprueba2')
if __name__ == "__main__":
    main(sys.argv[1:])





