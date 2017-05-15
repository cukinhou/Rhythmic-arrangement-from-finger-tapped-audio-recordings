import os
import pandas as pd
from essentia.standard import MonoLoader
from feature_extractor import FeatureExtractor


class Dataset(object):
    DEFAULT_PATH = '../Dataset/'

    def __init__(self, path=DEFAULT_PATH):

        assert os.path.isdir(path + 'EP') and os.path.isdir(path + 'EP'), \
            'Error creating dataset: there must be a EP and NEP folders in the root directory'

        self.path = path

    def generate(self):

        data = pd.DataFrame()

        for expertise_level in ['EP', 'NEP']:
            for root, dirs, files in os.walk(self.path+expertise_level):

                for file in files:
                    if file.endswith('.WAV') or file.endswith('.wav'):
                        audio = MonoLoader(filename=root+'/'+file)()
                        data = data.append(FeatureExtractor().feature_extractor(audio, file, expertise_level))

        return data

    def save(self, data, path=DEFAULT_PATH, name='default.csv'):
        if name.endswith('.csv'):
            data.to_csv(path+'/'+name, index_label='name')
        elif name.endswith('.json'):
            data.to_json(path+'/'+name)
        elif name.endswith('.arff'):
            print 'Currently you cannot save ARFF files'

if __name__ == '__main__':
    data = Dataset()
    data.save(data.generate())
