from ComparerBase import ComparerBase
from TrainComparing import TrainComparing
from TestComparing import TestComparing

class MeanComparer(ComparerBase):
    
    def analyse_data(self, full_data):
        if 'is_duplicate' in full_data.columns:
            self.mean = full_data['is_duplicate'].mean()
        else:
            self.mean = 0.36919785302629299

    def predict_pair(self, id, question1, question2):
        return self.mean

print(TrainComparing().get_score(MeanComparer()))
#TestComparing().predict_and_write_to_csv(MeanComparer())