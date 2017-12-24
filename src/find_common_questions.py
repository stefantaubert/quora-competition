# alle Fragen auflisten die es gibt und mit Frage als Tupel speichern in csv Datei. muss id enthalten
# Features berechnen.
# Prediction dafür ausführen.
# Top 5 der größten Ergebnisse auflisten.
from time import clock

import FeatureEngineering
import Preprocessing
import pandas as pd

import Paths
import prediction


def get_all_questions():
    # q_dict = defaultdict(set)
    df_test = pd.read_csv(Paths.Get_TEST_PREPROCESSED_Path(), encoding="ISO-8859-1")
    df_test.fillna('', inplace=True)
    df_train = pd.read_csv(Paths.Get_TRAIN_PREPROCESSED_Path(), encoding="ISO-8859-1")
    df_train.fillna('', inplace=True)
    
    ques = pd.concat([df_train['question1'], df_train['question2'], df_test['question1'], df_test['question2']], axis=0).reset_index(drop='index')

    print(ques.shape)

    ques = ques.unique()[:2000000] # bei allen reicht der ram nicht aus

    return ques 

def generate_data(question):
    qs = get_all_questions()
    df = pd.DataFrame()
    df['test_id'] = range(0, len(qs))
    df['question1'] = qs 
    df['question2'] = question
    #print(df)
    df.to_csv(Paths.Get_TMP_DATA_Path(), index=False)

def get_top_questions(count):
    submission = pd.read_csv(Paths.Get_TMP_SUBMISSION_Path(), encoding="ISO-8859-1")
    result = submission.sort_values(by=['is_duplicate'], ascending=[False])
    original_data = pd.read_csv(Paths.Get_TMP_DATA_Path(), encoding="ISO-8859-1")
    print(result[:5])
    counter = 0
    for index, row in result.iterrows():
        print(original_data.question1[index])
        counter = counter + 1
        if counter == count:
            return

start = clock()

generate_data("how can i make money through the internet")
Preprocessing.execute(4) 
FeatureEngineering.execute(4)
prediction.execute(4)
get_top_questions(10)

print('Overall duration: ', round(clock()-start, 2), 'seconds')