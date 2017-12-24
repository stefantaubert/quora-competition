from time import clock
import features.TfIdfFeatures as tfidf
import features.WordLengthFeatures as wl
import features.SharedWordsFeature as sw
import features.FrequencyFeatures as intersection
import features.LevenshteinFeature as lev
import Paths
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion

# pipeline = Pipeline([
#   ('extract_essays', EssayExractor()),
#   ('features', FeatureUnion([
#     ('ngram_tf_idf', Pipeline([
#       ('counts', CountVectorizer()),
#       ('tf_idf', TfidfTransformer())
#     ])),
#     ('essay_length', LengthTransformer()),
#     ('misspellings', MispellingCountTransformer())
#   ])),
#   ('classifier', MultinomialNB())
# ])

def foo():
    data = pd.read_csv(Paths.Get_TRAIN_PREPROCESSED_Path(), encoding="ISO-8859-1")
    data.fillna('', inplace=True)

    comb_features = FeatureUnion({('lev', lev.LevenshteinExtractor())})
    all_features = comb_features.transform(data)
    print(all_features)

foo()


def __engineer_features__(data_in, data_out, features):
    start = clock()
    data = pd.read_csv(data_in, encoding="ISO-8859-1")
    data.fillna('', inplace=True)
    print(data.shape)

    for feature in features:
        feature.apply(data)
    print(data.shape)
    data.to_csv(data_out, index=False)
    print('Duration: ', round(clock()-start, 2), 'seconds')

def actualise_features(mode=1, features = [], root='/media/stefan/main-data/my-uni-files/VI/bachelorarbeit/data/'):
    Paths.init(root)
    if mode == 1 or mode == 3:
        __engineer_features__(Paths.Get_TRAIN_FEATURES_Path(), Paths.Get_TRAIN_FEATURES_Path(), features)
        
    if mode == 2 or mode == 3:
        __engineer_features__(Paths.Get_TEST_FEATURES_Path(), Paths.Get_TEST_FEATURES_Path(), features)

#1 = Train, 2 = Test, 3 = Beides, 4 = tmp
def execute(mode=1, root='/media/stefan/main-data/my-uni-files/VI/bachelorarbeit/data/'):
    '''Ließt die vorverarbeiteten Daten ein und extrahiert alle Features.'''
    start = clock()
    Paths.init(root)
    intersection.init(Paths)
    print("Init in: " + str(clock() - start) + "seconds")
 
    features = [wl, sw, tfidf, intersection, lev]

    if mode == 1 or mode == 3:
        __engineer_features__(Paths.Get_TRAIN_PREPROCESSED_Path(), Paths.Get_TRAIN_FEATURES_Path(), features)

    if mode == 2 or mode == 3:
        __engineer_features__(Paths.Get_TEST_PREPROCESSED_Path(), Paths.Get_TEST_FEATURES_Path(), features)

    if mode == 4:
        __engineer_features__(Paths.Get_TMP_PREPROCESSED_Path(), Paths.Get_TMP_FEATURES_Path(), features)