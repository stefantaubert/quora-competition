import functools
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from .NoFitMixin import NoFitMixin
import time
from .Tokenizer import Tokenizer


class SharedWordsExtractor(NoFitMixin):

    # Quelle: http://love-python.blogspot.de/2012/07/python-code-to-compute-jaccard-index.html
    def compute_jaccard_index_optimized(self, set_1, set_2):
        if set_1 or set_2:
            n = len(set_1.intersection(set_2))
            return n / float(len(set_1) + len(set_2) - n)
        else:
            return 1.0


    # print(compute_jaccard_index_optimized(set("How I can speak English fluently?".split()), set("How can I learn to speak English fluently?".split())))
    # print(compute_jaccard_index_optimized(set(), set()))


    def jaccard(self, row):
        return self.compute_jaccard_index_optimized(set(self.tok.split(row[0])), set(self.tok.split(row[1])))


    def word_match_share(self, row, stops=[]):
        q1words = {}
        q2words = {}

        for word in self.tok.split(row[0]):
            if word not in stops:
                q1words[word] = 1
        for word in self.tok.split(row[1]):
            if word not in stops:
                q2words[word] = 1
        if len(q1words) == 0 or len(q2words) == 0:
            # The computer-generated chaff includes a few questions that are nothing but stopwords
            return 0

        shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
        shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
        R = (len(shared_words_in_q1) + len(shared_words_in_q2)) / (len(q1words) + len(q2words))
        return R

    def __init__(self):
        self.tok = Tokenizer()

    def transform(self, data):
        start = time.time()
        result = pd.DataFrame()
        stops = set(stopwords.words("english"))
        f1 = functools.partial(self.word_match_share, stops=stops)

        result['jaccard'] = data.apply(self.jaccard, axis=1, raw=True)
        result['common_words'] = data.apply(lambda x: len(set(self.tok.split(x[0])).intersection(set(self.tok.split(x[1])))), axis=1, raw=True)
        result['word_match'] = data.apply(self.word_match_share, axis=1, raw=True)
        result['word_match_stops'] = data.apply(f1, axis=1, raw=True)

        print("Duration " + self.__class__.__name__ + ": " + str(time.time() - start))
        return result

