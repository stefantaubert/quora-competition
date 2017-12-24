from comparing_base import ComparingBase

class WordMatchComparer(ComparingBase):
    def predict_duplicate(self, question1, question2, is_duplicate):
        return word_match_share(question1,  question2)

class WordMatchWithoutStopwordsComparer(ComparingBase):
    def predict_duplicate(self, question1, question2, is_duplicate):
        return word_match_share_without_stopwords(question1,  question2)

#WordMatchWithoutStopwords().full_run()
#test_run(WordMatchWithoutStopwords)
#Predicted score: 0.755722736352
#Time:  1.97 seconds

#full_run(WordMatchWithoutStopwords)
#Predicted score: 0.759205343258
#Time:  36.98 seconds

from string_comparer import * 
from string_cleaner import *

class WordMatchComparerCleaned(ComparingBase):
    def predict_duplicate(self, question1, question2, is_duplicate):
        q1_clean = unify_text(question1)
        q2_clean = unify_text(question2)
        return word_match_share(q1_clean,  q2_clean)

#test_run(WordMatchComparerCleaned)

#Predicted score: 0.621350746243
#Time:  6.59 seconds
from collections import Counter
import pandas as pd 
import numpy as np
class TfIdfComparer(ComparingBase):

    # If a word appears only once, we ignore it completely (likely a typo)
    # Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
    def get_weight(self, count, eps=10000, min_count=2):
        if count < min_count:
            return 0
        else:
            return 1 / (count + eps)

    def __init__(self):
        ComparingBase.__init__(self)
        train_qs = pd.Series(self.df_train['question1'].tolist() + self.df_train['question2'].tolist()).astype(str)
        #train_qs = train_qs.unique() # bringt 0.15 Verschlecherung
        self.eps = 5000
        words = (" ".join(train_qs)).lower().split()
        counts = Counter(words)
        self.weights = {word: self.get_weight(count) for word, count in counts.items()}

    def tfidf_word_match_share(self, question1, question2):
        q1words = {}
        q2words = {}
        for word in str(question1).lower().split():
            q1words[word] = 1
        for word in str(question2).lower().split():
            q2words[word] = 1

        shared_weights = [self.weights.get(w, 0) for w in q1words.keys() if w in q2words] + [self.weights.get(w, 0) for w in q2words.keys() if w in q1words]
        total_weights = [self.weights.get(w, 0) for w in q1words] + [self.weights.get(w, 0) for w in q2words]

        R = np.sum(shared_weights) / np.sum(total_weights)
        return R

    def predict_duplicate(self, question1, question2, is_duplicate):
        return self.tfidf_word_match_share(question1, question2)

TfIdfComparer().full_run()
