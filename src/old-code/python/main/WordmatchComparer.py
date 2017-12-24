from ComparerBase import ComparerBase
from TrainComparing import TrainComparing
from stanford_tagger import stanford_tokenizer

use_stanford = True

def word_match_share(question1, question2):
    q1words = {}
    q2words = {}

    if use_stanford:
        words1 = stanford_tokenizer.tokenize(question1)
        words2 = stanford_tokenizer.tokenize(question2)
    else:
        words1 = question1.split()
        words2 = question2.split()

    for word in words1:
        q1words[word] = 1
    for word in words2:
        q2words[word] = 1
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R

class WordMatchComparer(ComparerBase):
    
    def predict_pair(self, id, question1, question2):
        return word_match_share(question1, question2)

print(TrainComparing().get_score(WordMatchComparer(), 5))