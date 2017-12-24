from ComparerBase import ComparerBase
from TrainComparing import TrainComparing
from TestComparing import TestComparing

class UnequalComparer(ComparerBase):
    
    def predict_pair(self, id, question1, question2):
        return 0

#print(TrainComparing().get_score(UnequalComparer()))
TestComparing().predict_and_write_to_csv(UnequalComparer())