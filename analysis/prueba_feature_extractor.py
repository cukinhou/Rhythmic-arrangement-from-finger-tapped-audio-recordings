from feature_extractor import FeatureExtractor
import pandas as pd
import essentia.standard as es
from os import listdir
import numpy as np

path = '../Dataset/'
list_files = listdir(path)
list_files = [file for file in list_files if (file.endswith('.WAV') or file.endswith('.wav'))]
dataset = pd.DataFrame()
shape = 0
for file in list_files:

    extractor = FeatureExtractor()
    audio = es.MonoLoader(filename=path + file)()
    features = extractor.feature_extractor(audio, file, 'ep')
    dataset = dataset.append(features)

print dataset
