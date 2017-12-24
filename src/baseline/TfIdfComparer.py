from collections import Counter
import pandas as pd
import numpy as np
from ComparerBase import ComparerBase
from TrainComparing import TrainComparing
from TestComparing import TestComparing
from stanford_tagger import stanford_tokenizer

class TfIdfComparer(ComparerBase):

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

    # If a word appears only once, we ignore it completely (likely a typo)
    # Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
    def get_weight(self, count, eps=10000, min_count=2):
        if count < min_count:
            return 0
        else:
            return 1 / (count + eps)

    def analyse_data(self, complete_data):
        all_questions = pd.Series(complete_data['question1'].tolist() + complete_data['question2'].tolist()).astype(str)
        #train_qs = train_qs.unique() # bringt 0.15 Verschlecherung
        self.eps = 5000

        words = []
        #for qs in all_questions:
        #    words.append(stanford_tokenizer.tokenize(qs))

        words = (" ".join(all_questions)).lower().split()
        counts = Counter(words)
        self.weights = {word: self.get_weight(count) for word, count in counts.items()}

    def predict_pair(self, id, question1, question2):
        return self.tfidf_word_match_share(question1, question2)

#print(TrainComparing().get_score(TfIdfComparer()))
TestComparing().predict_and_write_to_csv(TfIdfComparer())