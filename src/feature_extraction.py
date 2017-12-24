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


def extract_features(root, include_test, make_backup=False):
    now = time.time()
    Paths.init(root)

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
    train_data = pd.read_csv(Paths.Get_TRAIN_DATA_Path())
    train_features = pipeline.transform(train_data)
    train_features.to_csv(Paths.Get_TRAIN_FEATURES_Path(), index=False)

    if make_backup:
        train_features.to_csv(Paths.Get_TRAIN_FEATURES_BACKUP_Path(), index=False)

    print('train feature extraction duration: ' + str(time.time() - now))

    if include_test:
        # Features für die Test-Daten berechnen
        test_data = pd.read_csv(Paths.Get_TEST_DATA_Path())
        test_features = pipeline.transform(test_data)
        test_features.to_csv(Paths.Get_TEST_FEATURES_Path(), index=False)

        if make_backup:
            test_features.to_csv(Paths.Get_TEST_FEATURES_BACKUP_Path(), index=False)

        print('total feature extraction duration: ' + str(time.time() - now))