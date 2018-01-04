from features.TfIdfFeatures import TfIdfExtractor
from features.WordLengthFeatures import WordLengthExtractor
from features.SharedWordsFeature import SharedWordsExtractor
from features.FrequencyFeatures import FrequencyExtractor
from features.LevenshteinFeature import LevenshteinExtractor
import data_paths
import pandas as pd
from sclearn_helper import DFFeatureUnion, DFTransform
from sklearn.pipeline import Pipeline
import time

def get_features(path):
    pipeline = Pipeline([
        ('preprocessing', Pipeline([
            ('fill_na', DFTransform(lambda X: X.fillna('NA'))),
            ('extract_q1_q2', DFFeatureUnion([
                ('q1', Pipeline([
                    ('select', DFTransform(lambda X: X['question1'])),
                    ('lower', DFTransform(lambda X: X.str.lower())),
                ])),
                ('q2', Pipeline([
                    ('select', DFTransform(lambda X: X['question2'])),
                    ('lower', DFTransform(lambda X: X.str.lower())),
                ])),
            ]))
        ])),
        ('extract_features', DFFeatureUnion([
            ('levenshtein', LevenshteinExtractor()),
            ('frequency', FrequencyExtractor()),
            ('shared_words', SharedWordsExtractor()),
            ('word_lengths', WordLengthExtractor()),
            ('tf_idf', TfIdfExtractor()),
        ], n_jobs=-1))  # Multicore-Unterstützung
    ])

    # Features für die Trainings-Daten berechnen
    data = pd.read_csv(path, encoding="ISO-8859-1")
    features = pipeline.transform(data)
    return features


def extract_features(include_test, make_backup=False):
    now = time.time()

    train_features = get_features(data_paths.train)
    train_features.to_csv(data_paths.train_features, index=False)
    if make_backup:
        train_features.to_csv(data_paths.train_features_backup, index=False)

    print('train feature extraction duration: ' + str(time.time() - now))

    if include_test:
        # Features für die Test-Daten berechnen
        test_features = get_features(data_paths.test)
        test_features.to_csv(data_paths.test_features, index=False)
#        if make_backup:
#            test_features.to_csv(data_paths.test_features_backup, index=False)

    print('total feature extraction duration: ' + str(time.time() - now))