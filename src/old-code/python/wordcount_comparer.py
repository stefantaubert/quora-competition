from comparing_base import *
from string_comparer import * 

class WordCountComparer(ComparingBase):
    def predict_duplicate(self, question1, question2, is_duplicate):
        equal_score = get_question_equal_score(question1, question2)
        equal_terms_count = equal_score[0]
        term_count = equal_score[1]

        return equal_terms_count / term_count

#test_run(WordCountComparer)
#Predicted score: 0.952151860156
#Time:  17.91 seconds

#full_run(WordCountComparer)

class NoStoppwordComparer(ComparingBase):
    def predict_duplicate(self, question1, question2, is_duplicate):
        q1_clean = remove_stoppwords(question1)
        q2_clean = remove_stoppwords(question2)
        equal_score = get_question_equal_score(q1_clean, q2_clean)
        equal_terms_count = equal_score[0]
        term_count = equal_score[1]

        if term_count == 0:
            return 0
        else:
            return equal_terms_count / term_count

#test_run(NoStoppwordComparer)
#Predicted score: 1.39440647854
#Time:  136.37 seconds

#full_run(NoStoppwordComparer)
#Predicted score: 1.26599295716
#Time:  2334.38 seconds

class UnifiedComparer(ComparingBase):
    def predict_duplicate(self, question1, question2, is_duplicate):
        #q1_clean = unify_text(question1)
        #q2_clean = unify_text(question2)
        equal_score = get_question_equal_score(question1, question2)
        equal_terms_count = equal_score[0]
        term_count = equal_score[1]

        if term_count == 0:
            return 0
        else:
            return equal_terms_count / term_count

#test_run(UnifiedComparer)
#Predicted score: 0.976613368707
#Time:  31.49 seconds

#full_run(NoStoppwordComparer)