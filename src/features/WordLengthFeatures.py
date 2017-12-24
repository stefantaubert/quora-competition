from nltk.corpus import stopwords
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from .NoFitMixin import NoFitMixin
import time
from .Tokenizer import Tokenizer

SAFE_DIFF = 0.0001

class WordLengthExtractor(NoFitMixin):
    def __init__(self):
        self.tok = Tokenizer()

    def transform(self, data):
        start = time.time()
        result = pd.DataFrame()
        stops = set(stopwords.words("english"))

        q1 = data.question1.map(lambda x: self.tok.split(x))
        q2 = data.question2.map(lambda x: self.tok.split(x))
        data_split = pd.concat([q1, q2], axis=1)

        q1_stops = q1.apply(lambda x: [y for y in x if y not in stops])
        q2_stops = q2.apply(lambda x: [y for y in x if y not in stops])

        data_split_stops = pd.concat([q1_stops, q2_stops], axis=1)

        #Feature: Length of Question
        result['len_q1'] = data.question1.apply(len)#
        result['len_q2'] = data.question2.apply(len)#
        result['min_len'] = data.apply(lambda x: min(len(x[0]), len(x[1])), axis=1, raw=True)
        result['max_len'] = data.apply(lambda x: max(len(x[0]), len(x[1])), axis=1, raw=True)

        #Feature: Difference in length between the Questions
        result['len_diff'] = result.max_len - result.min_len
        result['len_ratio'] = result.max_len / result.min_len + (SAFE_DIFF)

        #Feature: Character count of Question
        result['len_q1_blank'] = data.question1.apply(lambda x: len(x.replace(' ', '')))#
        result['len_q2_blank'] = data.question2.apply(lambda x: len(x.replace(' ', '')))#
        result['min_len_blank'] = data.apply(lambda x: min(len(x[0].replace(' ', '')), len(x[1].replace(' ', ''))), axis=1, raw=True)
        result['max_len_blank'] = data.apply(lambda x: max(len(x[0].replace(' ', '')), len(x[1].replace(' ', ''))), axis=1, raw=True)
        result['len_diff_blank'] = result.max_len_blank - result.min_len_blank
        result['len_ratio_blank'] = result.max_len_blank / (result.min_len_blank + SAFE_DIFF)

        #Feature: Word count of Question
        # Teil 1
        result['count_word_q1'] = q1.apply(len)#
        result['count_word_q2'] = q2.apply(len)#
        result['min_word_count'] = data_split.apply(lambda x: min(len(x[0]), len(x[1])), axis=1, raw=True)
        result['max_word_count'] = data_split.apply(lambda x: max(len(x[0]), len(x[1])), axis=1, raw=True)
        result['count_words_diff'] = result.max_word_count - result.min_word_count
        result['count_words_ratio'] = result.max_word_count / (result.min_word_count + SAFE_DIFF)
        result['total_words'] = result.min_word_count + result.max_word_count

        # Teil 2
        result['min_unique_word_count'] = data_split.apply(lambda x: min(len(set(x[0])), len(set(x[1]))), axis=1, raw=True)
        result['max_unique_word_count'] = data_split.apply(lambda x: max(len(set(x[0])), len(set(x[1]))), axis=1, raw=True)
        result['len_words_q1_unique'] = q1.apply(lambda x: len(set(x)))#
        result['len_words_q2_unique'] = q2.apply(lambda x: len(set(x)))#
        result['diff_unique_word_count'] = result.max_unique_word_count - result.min_unique_word_count
        result['ratio_unique_word_count'] = result.max_unique_word_count / (result.min_unique_word_count + SAFE_DIFF)
        result['total_unique_words'] = data_split.apply(lambda x: len(set(x[0]).union(set(x[1]))), axis=1, raw=True)

        # Teil 3
        result['len_words_q1_unique_no_stops'] = q1_stops.apply(lambda x: len(set(x)))#
        result['len_words_q2_unique_no_stops'] = q2_stops.apply(lambda x: len(set(x)))#
        result['min_unique_nostops_word_count'] = data_split_stops.apply(lambda x: min(len(set(x[0])), len(set(x[1]))), axis=1, raw=True)
        result['max_unique_nostops_word_count'] = data_split_stops.apply(lambda x: max(len(set(x[0])), len(set(x[1]))), axis=1, raw=True)
        result['count_words_diff_unique_no_stops'] = result.max_unique_nostops_word_count - result.min_unique_nostops_word_count
        result['count_words_ratio_unique_no_stops'] = result.max_unique_nostops_word_count / (result.min_unique_nostops_word_count + SAFE_DIFF)
        result['total_unique_no_stopwords'] = data_split_stops.apply(lambda x: len(set(x[0]).union(set(x[1]))), axis=1, raw=True)

        #Teil 4
        data['char_diff_unique_no_stops'] = data_split_stops.apply(lambda x: abs(len(''.join(x[0])) - len(''.join(x[1]))), axis=1, raw=True)
        #data['same_start_word'] = data_split.apply(lambda x: 0 if not x[0] or not x[1] else int(x[0][0] == x[1][0]), axis=1, raw=True)

        print("Duration " + self.__class__.__name__ + ": " + str(time.time() - start))
        return result
