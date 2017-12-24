from ComparerBase import ComparerBase
from TrainComparing import TrainComparing
from TestComparing import TestComparing
from random import random

class RandomComparer(ComparerBase):
    
    def predict_pair(self, id, question1, question2):
        nr = random() # Version 1
        #nr = round(nr, 0) # Version 2
        #print(nr)
        return nr

#print(TrainComparing().get_score(RandomComparer()))
TestComparing().predict_and_write_to_csv(RandomComparer())