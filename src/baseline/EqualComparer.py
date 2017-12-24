from ComparerBase import ComparerBase
from TrainComparing import TrainComparing
from TestComparing import TestComparing

class EqualComparer(ComparerBase):

    def predict_pair(self, id, question1, question2):
        return 1

print(TrainComparing().get_score(EqualComparer()))
#TestComparing().predict_and_write_to_csv(EqualComparer())