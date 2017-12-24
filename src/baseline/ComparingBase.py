from pandas import read_csv
from time import * # tracking execution times

class ComparingBase:

    def __init__(self, path_to_csv):
        self.data = read_csv(path_to_csv)
        self.data_size = len(self.data.index)

    def analyse_row(self, id, row):
        pass

    def predict_pairs(self, comparer, percent_of_data):
        comparer.analyse_data(self.data)
        question_count = round(self.data_size / 100 * percent_of_data)
        print("Predict pairs for {} of {} questions...".format(question_count, self.data_size))
 
        predicted_values = [0]*question_count
        show_process_steps = round(question_count / 4)

        for index, row in self.data.iterrows():
            all_data_compared = index >= question_count

            if all_data_compared:
                break

            is_time_to_show_process = index % show_process_steps == 0

            if is_time_to_show_process:
                print('Processed {} out of {}'.format(index, question_count))
            
            first_question = str(row['question1'])
            second_question = str(row['question2'])

            self.analyse_row(id, row)
            predicted = comparer.predict_pair(index, first_question, second_question)

            predicted_values[index] = predicted
        
        return predicted_values
