from features.TfIdfFeatures import TfIdfExtractor
from features.WordLengthFeatures import WordLengthExtractor
from features.SharedWordsFeature import SharedWordsExtractor
from features.FrequencyFeatures import FrequencyExtractor
from features.LevenshteinFeature import LevenshteinExtractor
import Paths
import pandas as pd
from sclearn_helper import DFFeatureUnion, DFTransform
from sklearn.pipeline import Pipeline
import time
import sys

if __name__ == '__main__':
    now = time.time()

    # Root-Verzeichnis aus Parametern lesen und Pfade initialisieren
    root = sys.argv[1]
    Paths.init(root)

    pipeline = Pipeline([
        ('preprocessing', Pipeline([
            ('fill_na', DFTransform(lambda X: X.fillna('NA'))),
            ('extract_q1_q2', DFFeatureUnion([
                ('q1', Pipeline([
                    ('select', DFTransform(lambda X: X['question1'])),
                    #('lower', DFTransform(lambda X: X.str.lower())),
                ])),
                ('q2', Pipeline([
                    ('select', DFTransform(lambda X: X['question2'])),
                    #('lower', DFTransform(lambda X: X.str.lower())),
                ])),
            ]))
        ])),
        ('extract_features', DFFeatureUnion([
            ('levenshtein', LevenshteinExtractor()),
            ('frequency', FrequencyExtractor()),
            ('shared_words', SharedWordsExtractor()),
            ('word_lengths', WordLengthExtractor()),
            ('tf_idf', TfIdfExtractor()),
        ], n_jobs=-1)) # Multicore-Unterstützung
    ])

    train_data = pd.read_csv(Paths.Get_TRAIN_DATA_Path())
    train_features = pipeline.transform(train_data)
    train_features.to_csv(Paths.Get_TRAIN_FEATURES_Path(), index=False)

    print('train feature extraction duration: ' + str(time.time()-now))

    test_data = pd.read_csv(Paths.get_test_data_path())
    test_features = pipeline.transform(test_data)
    test_features.to_csv(Paths.Get_TEST_FEATURES_Path(), index=False)

    print('total feature extraction duration: ' + str(time.time()-now))

    #print(res)

