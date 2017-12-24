from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np 
import pandas as pd
from itertools import chain
import functools
from nltk.corpus import stopwords
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from .NoFitMixin import NoFitMixin
from .Tokenizer import Tokenizer
import time
SAFE_DIFF = 0.0001


class TfIdfExtractor(NoFitMixin):

    def tfidf_word_match_share(self, row, weights=None):
        q1words = {}
        q2words = {}
        for word in self.tok.split(row[0]):
            q1words[word] = 1
        for word in self.tok.split(row[1]):
            q2words[word] = 1
        if len(q1words) == 0 or len(q2words) == 0:
            # The computer-generated chaff includes a few questions that are nothing but stopwords
            return 0

        shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
        total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

        R = np.sum(shared_weights) / (np.sum(total_weights) + SAFE_DIFF)
        return R


    def tfidf_word_match_share_stops(self, row, stops=None, weights=None):
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

        shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
        total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

        R = np.sum(shared_weights) / (np.sum(total_weights) + SAFE_DIFF)
        return R


    def get_weight(self, count, eps=10000, min_count=2):
        if count < min_count:
            return 0
        else:
            return 1 / (count + eps)

    def __init__(self):
        self.tok = Tokenizer()

    def transform(self, data):
        start = time.time()
        result = pd.DataFrame()

        stops = set(stopwords.words("english"))

        all_questions = pd.Series(data.question1.tolist() + data.question2.tolist())
        all_questions = all_questions.map(lambda x: self.tok.split(x))

        words = list(chain.from_iterable(all_questions))
        counts = Counter(words)
        weights = {word: self.get_weight(count) for word, count in counts.items()}
        f = functools.partial(self.tfidf_word_match_share, weights=weights)

        result['tfidf_wm'] = data.apply(f, axis=1, raw=True)

        f = functools.partial(self.tfidf_word_match_share_stops, stops=stops, weights=weights)
        result['tfidf_wm_stops'] = data.apply(f, axis=1, raw=True)

        print("Duration " + self.__class__.__name__ + ": " + str(time.time() - start))
        return result

        #all_questions = pd.Series(data.question1.tolist() + data.question2.tolist())
        #total_words = [item for sublist in all_questions for item in sublist]
        #str_q = all_questions.apply(lambda x: ' '.join(y) for y in x)
        #print(str_q[:10])
        # total_words = list(chain.from_iterable(all_questions))
        #print(total_words[:10])

        #vectorizer = TfidfVectorizer(stop_words = 'english')
        #vectorizer.fit(df_train['question1'] + df_train['question2'])
        #total_words = list(set(vectorizer.get_feature_names()))

        #vectorizer = TfidfVectorizer(stop_words = 'english', vocabulary = total_words)
        #vectorizer.fit(all_questions)
        #tf_diff = vectorizer.transform(data.question1) - vectorizer.transform(data.question2)
        #return tf_diff