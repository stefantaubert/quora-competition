from ComparerBase import ComparerBase
from TrainComparing import TrainComparing
from TestComparing import TestComparing
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def match_score(arr1, arr2):
    q1words = {}
    q2words = {}
    for word in arr1:
        q1words[word] = 1
    for word in arr2:
        q2words[word] = 1
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R

class PorterStemmingComparer(ComparerBase):

    stemmed_words_cache = {}

    def predict_pair(self, id, question1, question2):
        stemmed_words_1 = []
        for word in question1.split():
            if not self.stemmed_words_cache.__contains__(word):
                try:
                    self.stemmed_words_cache[word] = stemmer.stem(word)
                except:
                    print("Error at word:" + word)
                    raise Exception('I know Python!')

            stemmed_words_1.append(self.stemmed_words_cache[word])

        stemmed_words_2 = []
        for word in question2.split():
            if not self.stemmed_words_cache.__contains__(word):
                try:
                    self.stemmed_words_cache[word] = stemmer.stem(word)
                except:
                    print("Error at word:" + word)
                    raise Exception('I know Python!')

            stemmed_words_2.append(self.stemmed_words_cache[word])

        return match_score(stemmed_words_1, stemmed_words_2)

def Test():
    s1 = "Why is the USA the most powerful country of the world".split()
    s2 = "Is USA the most powerful country of the world?".split()

    singles = [stemmer.stem(word) for word in s1]
    print(' '.join(singles))

    singles = [stemmer.stem(word) for word in s2]
    print(' '.join(singles))

#print(TrainComparing().get_score(PorterStemmingComparer()))
TestComparing().predict_and_write_to_csv(PorterStemmingComparer())