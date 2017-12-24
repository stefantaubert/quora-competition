import Paths
import pandas as pd 
import

def process(root):
    start =
    Paths.init(root)
    s_1 = 0.1
    s_0 = 0.2

    submission = pd.read_csv(Paths.Get_SUBMISSION_Path(), encoding="ISO-8859-1")
    submission.is_duplicate = submission.is_duplicate.apply(lambda x: 1 if x >= 1-s_1 else x)
    submission.is_duplicate = submission.is_duplicate.apply(lambda x: 0 if x <= s_0 else x)
    submission.to_csv(Paths.Get_SUBMISSION_Path(), index=False)
