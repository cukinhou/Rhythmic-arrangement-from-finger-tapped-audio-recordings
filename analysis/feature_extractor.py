import essentia.standard as es
import numpy as np
from essentia import array
import pandas as pd
from madmom.features.onsets import  CNNOnsetProcessor, peak_picking
from feature_keys import  low_level_keys

class FeatureExtractor(object):
    def __init__(self, onset_wise=True, frame_size=2048, hop_size=1024, sr=44100.):
        self.onset_wise = onset_wise
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.sr = sr

    def onset_detector(self, audio):

        onset_strength = CNNOnsetProcessor()(audio)
        onset_frames = peak_picking(onset_strength, threshold=0.5)
        frame_rate = len(audio) / len(onset_strength)
        onset_times = onset_frames * frame_rate
        onset_end_times = np.append(onset_times[1:], len(audio))

        return onset_times, onset_end_times

    def frame_feature(self, feature, onset_times, onset_end_times, frame_rate):

        return map(
            lambda x, y: np.mean(
                feature[int(np.ceil(x * frame_rate)):int(np.ceil(y * frame_rate))]
            ), onset_times, onset_end_times
        )

    def onset_slicer(self, audio, onset_start_times, onset_end_times):
            for t in range(0, len(onset_start_times)):
                yield audio[int(onset_start_times[t]*self.sr) : int(onset_end_times[t]*self.sr)]

    def feature_extractor(self, audio, file_name, label):

        features = es.LowLevelSpectralExtractor(
            frameSize=self.frame_size, hopSize=self.hop_size, sampleRate=self.sr
        )(audio)

        out_data = pd.DataFrame(columns=low_level_keys)

        onset_start_times, onset_end_times = self.onset_detector(audio)

        n_feat = 0
        for feature in features:
            if any(isinstance(f, np.ndarray) for f in feature):
                for i in feature.T:

                    framed_feature = self.frame_feature(i, onset_start_times, onset_end_times, 1./self.hop_size)
                    out_data[low_level_keys[n_feat]] = framed_feature / max(framed_feature)
                    n_feat += 1
            else:

                framed_feature = self.frame_feature(feature, onset_start_times, onset_end_times, 1./self.hop_size)
                out_data[low_level_keys[n_feat]] = framed_feature / max(framed_feature)
                n_feat += 1

        out_data.index = range(1, len(onset_start_times) + 1)
        out_data = out_data.T.add_prefix('file_'+file_name[:-4]+'_onset_')
        out_data = out_data.T
        out_data['label'] = label

        return out_data