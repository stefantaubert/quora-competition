import pandas as pd
from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin
from .NoFitMixin import NoFitMixin
import time
#
# q_dict = defaultdict(set)
#
#
# def read_neighbours(paths):
#     df_test = pd.read_csv(paths.getTEST_PREPROCESSED, encoding="ISO-8859-1")
#     df_test.fillna('', inplace=True)
#     df_train = pd.read_csv(paths.TRAIN_PREPROCESSED, encoding="ISO-8859-1")
#     df_train.fillna('', inplace=True)
#
#     ques = pd.concat([df_train[['question1', 'question2']], df_test[['question1', 'question2']]],
#                      axis=0).reset_index(drop='index')
#
#     for i in range(ques.shape[0]):
#         q_dict[ques.question1[i]].add(ques.question2[i])
#         q_dict[ques.question2[i]].add(ques.question1[i])
#
#
# def intersect(row):
#     return len(set(q_dict[row[0]]).intersection(set(q_dict[row[1]])))

class FrequencyExtractor(NoFitMixin):
    # def __init__(self, paths):
        # if len(q_dict) == 0:
        #     read_neighbours(paths)

    def transform(self, data):
        start = time.time()
        result = pd.DataFrame()
        q_dict = defaultdict(set)

        for i in range(data.shape[0]):
            q_dict[data.question1[i]].add(data.question2[i])
            q_dict[data.question2[i]].add(data.question1[i])

        result['intersect'] = data.apply(lambda x: len(set(q_dict[x[0]]).intersection(set(q_dict[x[1]]))), axis=1, raw=True)
        result['q1_freq'] = data.apply(lambda x: len(q_dict[x[0]]), axis=1, raw=True)
        result['q2_freq'] = data.apply(lambda x: len(q_dict[x[1]]), axis=1, raw=True)

        print("Duration " + self.__class__.__name__ + ": " + str(time.time() - start))
        return result


