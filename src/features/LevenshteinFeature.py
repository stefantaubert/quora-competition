from fuzzywuzzy import fuzz
import pandas as pd
from .NoFitMixin import NoFitMixin
from sklearn.base import BaseEstimator, TransformerMixin
import time
import editdistance

def test():
    #print(editdistance.eval("Does Russia's KGB still exist?", "What was Katje saying in the movie Bridge Of Spies?"))
    print(editdistance.eval("married", "unmarried"))
    print(fuzz.QRatio("married", "unmarried"))
    print(fuzz.WRatio("married", "unmarried"))
    print(fuzz.partial_ratio("married", "unmarried"))
    print(fuzz.partial_token_set_ratio("married", "unmarried"))
    print(fuzz.partial_token_sort_ratio("married", "unmarried"))
    print(fuzz.token_set_ratio("married", "unmarried"))
    print(fuzz.token_sort_ratio("married", "unmarried"))


class LevenshteinExtractor(NoFitMixin):
    def transform(self, data):
        start = time.time()
        result = pd.DataFrame()

        result['levenshtein'] = data.apply(lambda x: editdistance.eval(x[0], x[1]), axis=1)
        # dauert zu lange und ist nicht in ba
        # result['fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(x[0], x[1]), axis=1)
        # result['fuzz_WRatio'] = data.apply(lambda x: fuzz.WRatio(x[0], x[1]), axis=1)
        # result['fuzz_partial_ratio'] = data.apply(lambda x: fuzz.partial_ratio(x[0], x[1]), axis=1)
        # result['fuzz_partial_token_set_ratio'] = data.apply(lambda x: fuzz.partial_token_set_ratio(x[0], x[1]), axis=1)
        # result['fuzz_partial_token_sort_ratio'] = data.apply(lambda x: fuzz.partial_token_sort_ratio(x[0], x[1]), axis=1)
        # result['fuzz_token_set_ratio'] = data.apply(lambda x: fuzz.token_set_ratio(x[0], x[1]), axis=1)
        # result['fuzz_token_sort_ratio'] = data.apply(lambda x: fuzz.token_sort_ratio(x[0], x[1]), axis=1)

        print("Duration " + self.__class__.__name__ + ": " + str(time.time() - start))
        return result