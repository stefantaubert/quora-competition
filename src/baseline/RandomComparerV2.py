from ComparerBase import ComparerBase
from TrainComparing import TrainComparing
from TestComparing import TestComparing
from random import random

class RandomComparerV2(ComparerBase):
    
    def predict_pair(self, id, question1, question2):
        nr = random()
        return nr

#print(TrainComparing().get_score(RandomComparerV2()))
TestComparing().predict_and_write_to_csv(RandomComparerV2())