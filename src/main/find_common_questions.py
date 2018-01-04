# alle Fragen auflisten die es gibt und mit Frage als Tupel speichern in csv Datei. muss id enthalten
# Features berechnen.
# Prediction dafür ausführen.
# Top 5 der größten Ergebnisse auflisten.
from time import clock
import pandas as pd
from sklearn.model_selection import train_test_split
import settings
import data_paths
import prediction
import feature_extraction
import feature_selection
import model_training

def get_all_questions():
    x_train = pd.read_csv(data_paths.train, encoding="ISO-8859-1")
    x_train.fillna('', inplace=True)
    y_train = x_train.is_duplicate

    # Trainings-Set und Validierungs-Set erstellen. Das Validierungs-Set enthält 10% aller Trainings-Daten.
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=settings.validation_size, random_state=settings.seed_validation_split_training)

    ques = pd.concat([x_valid['question1'], x_valid['question2']], axis=0).reset_index(drop='index')
    print(ques.shape)
    ques = ques.unique() # bei allen reicht der ram nicht aus
    print(ques.shape)
    return ques 

def generate_tmp_data(question):
    qs = get_all_questions()
    df = pd.DataFrame()
    df['test_id'] = range(0, len(qs))
    df['question1'] = qs 
    df['question2'] = question
    #print(df)
    df.to_csv(data_paths.tmp, index=False)

def get_top_questions(count):
    submission = pd.read_csv(data_paths.tmp_submission, encoding="ISO-8859-1")
    result = submission.sort_values(by=['is_duplicate'], ascending=[False])
    original_data = pd.read_csv(data_paths.tmp, encoding="ISO-8859-1")
    print(result[:5])
    counter = 0
    for index, row in result.iterrows():
        print(original_data.question1[index])
        counter = counter + 1
        if counter == count:
            return

if __name__ == '__main__':
    best_iteration = 111
    question = "What are the best ways to loose weight?"
    prepaire_traindata = True
    #prepaire_traindata = False

    start = clock()
    if prepaire_traindata:
        #prepare traindata
        #feature_extraction.extract_features(False, True)
        feature_selection.select_features_at(best_iteration)
        model_training.train_and_save_model()

        generate_tmp_data(question)

        train_features = feature_extraction.get_features(data_paths.tmp)
        train_features.to_csv(data_paths.tmp_features, index=False)

        prediction.predict_and_write_data(data_paths.tmp, data_paths.tmp_features, data_paths.tmp_submission)

    get_top_questions(30)

    print('Overall duration: ', round(clock()-start, 0), 'seconds')