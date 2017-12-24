from ComparerBase import ComparerBase
from TrainComparing import TrainComparing
from TestComparing import TestComparing
import nltk
from nltk.classify import apply_features
from time import clock # tracking execution times

amount_for_score_calc = 0.9

def word_match_share(words1, words2):
    q1words = {}
    q2words = {}

    for word in words1:
        q1words[word] = 1
    for word in words2:
        q2words[word] = 1
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R

class NaiveBayesClassifierComparer(ComparerBase):
#0.6806500284449282
#0.6821671679594744
#0.7080002374544879
#0.7284015355390216
#0.7391364571790406
#0.7790485990185214
#0.8139207004872739

    '''Für die ersten 50% der Trainigs-Fragen wird die classify Funktion benutzt. Die Trainigs-Daten und Validierungsdaten sind im Verhältnis 1:1 aufgeteilt'''
    def question_features(self, q1, q2):
        result = {}
        #result['same_word_count'] = len(q1.split()) == len(q2.split())
        q1_split = q1.split()
        q2_split = q2.split()
        result['q1_word_count'] = len(q1_split)
        result['q2_word_count'] = len(q2_split)
        result['same_last_word'] = q1_split[-1] == q2_split[-1]
        result['q1_last_word'] = q1_split[-1]
        result['q2_last_word'] = q2_split[-1]
        result['word_match'] = word_match_share(q1_split, q2_split)
        result['q1_freq'] = len(self.q_dict[q1])
        result['q2_freq'] = len(self.q_dict[q2])
        #print(set(self.q_dict[q1]))
        #print(set(self.q_dict[q2]))
        #print(set(self.q_dict[q1]).intersection(set(self.q_dict[q2])))
        result['q1_q2_intersect'] = len(set(self.q_dict[q1]).intersection(set(self.q_dict[q2])))
        #result['avg_c_words'] = int((len(q1_split) + len(q2_split)) / 2)
        #result['q1_len'] = len(q1)
        #result['q2_len'] = len(q2)

        return result

    def analyse_data(self, full_data):
        ques = full_data[['question1', 'question2']].reset_index(drop='index')
        from collections import defaultdict
        self.q_dict = defaultdict(set)
        for i in range(ques.shape[0]):
            self.q_dict[ques.question1[i]].add(ques.question2[i])
            self.q_dict[ques.question2[i]].add(ques.question1[i])

        splitting_at = int(len(full_data.index) * amount_for_score_calc)
        t1 = clock()
        data_set = [(self.question_features(str(row["question1"]), str(row["question2"])), row["is_duplicate"]) for (index, row) in full_data.iterrows() if index >= splitting_at]
        t2 = clock()
        print('Time for feature extraction: ', round(t2-t1, 2), 'seconds')
        splitting_train_at = int(len(data_set) * 0.5)
        train_set = data_set[splitting_train_at:]
        devtest_set = data_set[:splitting_train_at]
        self.classifier = nltk.NaiveBayesClassifier.train(train_set)
        self.classifier.show_most_informative_features(30)
        print("Accuracy", nltk.classify.accuracy(self.classifier, devtest_set))

    def predict_pair(self, id, question1, question2):
        predicted = self.classifier.classify(self.question_features(question1, question2))
        #print(predicted)
        #raise Exception("reicht..")
        return predicted

print(TrainComparing().get_score(NaiveBayesClassifierComparer(), amount_for_score_calc * 100))
#TestComparing().predict_and_write_to_csv(NaiveBayesClassifierComparer())