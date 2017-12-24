from ComparingBase import ComparingBase
from sklearn.metrics import log_loss
from stanford_tagger import pos_tag
from stanford_tagger import ner_tag
from time import clock # tracking execution times
import nltk
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

class TrainComparing(ComparingBase):
    
    def __init__(self):
        ComparingBase.__init__(self, "D:/dev/Python/quora-bachelor-thesis/data/train.csv")
        self.original_values = self.data["is_duplicate"]
        self.analyse_data()

    def get_question_equal_score(self, question1, question2):
        q1_parts = question1.lower().split()
        q2_parts = question2.lower().split()
        equal_terms_count = 0
        term_count = 0
        if len(q1_parts) <= len(q2_parts):
            term_count = len(q1_parts)
            for str in q1_parts:
                if str in q2_parts:
                    equal_terms_count = equal_terms_count + 1
        else:
            term_count = len(q2_parts)
            for str in q2_parts:
                if str in q1_parts:
                    equal_terms_count = equal_terms_count + 1
                    
        return [equal_terms_count, term_count]

    count_analyse_method = 0
    count_analyse_wrong_marked = 0

    def word_count_features(self, q1, q2):
        return {'same_word_count' : len(q1.split()) == len(q2.split())}

    def analyse_data(self):
        return
        featuresets = [(self.word_count_features(row["question1"], row["question2"]), row["is_duplicate"]) for (index, row) in self.data.iterrows()]
        train_set = featuresets
        classifier = nltk.NaiveBayesClassifier.train(train_set)
        print(classifier.classify(self.word_count_features("What does manipulation mean?","What does manipulation means?")))

    def analyse_row(self, id, row):
        return
        first_question = str(row['question1'])
        second_question = str(row['question2'])
        is_duplicate = row['is_duplicate']

        if len(nltk.sent_tokenize(first_question)) == 1 and len(nltk.sent_tokenize(second_question)) == 1:
            #bestehen aus einem Satz
            tokens1 = nltk.word_tokenize(first_question)
            tokens2 = nltk.word_tokenize(second_question)
            if len(tokens1) == len(tokens2):
                result = True
                for q1, q2 in zip(tokens1, tokens2):

                  
                    #pos1 = nltk.pos_tag([q1])[0][1]
                    #pos2 = nltk.pos_tag([q2])[0][1]

                    #if pos1 != pos2:
                        #result = False  
                        #break

                    stem1 = stemmer.stem(q1)
                    stem2 = stemmer.stem(q2)
                    if stem1 != stem2:
                        result = False  
                        break
                    
                   #kommt auf 21,1085 falsche
                if result:
                    if is_duplicate == 0:
                        #print(row)
                        #print(ner_tag(first_question))
                        #print(ner_tag(second_question))
                        self.count_analyse_wrong_marked += 1
                    else:
                        self.count_analyse_method += 1
        return
        first_question = str(row['question1'])
        second_question = str(row['question2'])
        is_duplicate = row['is_duplicate']
        tags1 = pos_tag(first_question)
        tags2 = pos_tag(second_question)

        if len(tags1) == len(tags2) and is_duplicate == 0:
            print(tags1)
            print(tags2)

        scores = self.get_question_equal_score(first_question, second_question)
        equal_terms_count = scores[0]
        term_count = scores[1]
        if is_duplicate == 0 and equal_terms_count == term_count and len(first_question) == len(second_question):
            print(pos_tag(first_question))
            print(pos_tag(second_question))

    def get_score(self, comparer, for_percent_of_data = 100):
        t1 = clock()
        self.predicted_values = self.predict_pairs(comparer, for_percent_of_data)
        length = len(self.predicted_values)
        score = log_loss(self.original_values[:length], self.predicted_values)
        t2 = clock()
        print('Time: ', round(t2-t1, 2), 'seconds')
        print("Estimated Time full-set:", round((t2-t1)/length*self.data_size))
        print("Analysed Score pos: {} neg: {}".format(self.count_analyse_method, self.count_analyse_wrong_marked))
        return score
# 8 9

#TrainComparing()