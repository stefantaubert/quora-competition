from ComparerBase import ComparerBase
from TrainComparing import TrainComparing
from TestComparing import TestComparing
from random import random

class IsDuplicateComparer(ComparerBase):
    def analyse_data(self, full_data):
        if 'is_duplicate' in full_data.columns:
            self.duplicates = full_data['is_duplicate'].tolist()
        else:
            raise Exception("Not supported!")
    def predict_pair(self, id, question1, question2):
        return self.duplicates[id]

print(TrainComparing().get_score(IsDuplicateComparer()))