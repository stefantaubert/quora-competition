from ComparingBase import ComparingBase
from sklearn.metrics import log_loss
from stanford_tagger import pos_tag
from time import clock # tracking execution times
import nltk
import pandas as pd
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

class TestComparing(ComparingBase):
    
    def __init__(self):
        ComparingBase.__init__(self, "../data/test.csv")

    def predict_and_write_to_csv(self, comparer):
        t1 = clock()
        predicted_values = self.predict_pairs(comparer, 100)
        length = len(predicted_values)
        sub = pd.DataFrame()
        sub['test_id'] = self.data['test_id']
        sub['is_duplicate'] = predicted_values
        sub.to_csv('../data/submissions/result.csv', index=False) #Score: 0.35372
        t2 = clock()
        print('Time: ', round(t2-t1, 2), 'seconds')