# alle Fragen auflisten die es gibt und mit Frage als Tupel speichern in csv Datei. muss id enthalten
# Features berechnen.
# Prediction dafür ausführen.
# Top 5 der größten Ergebnisse auflisten.
from time import clock
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
import settings
import script_evaluation
import data_paths
import prediction
import xgboost as xgb
import numpy
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
    counts = Counter(ques)
    print(counts.most_common(50))

    return ques 

if __name__ == '__main__':
    #get_all_questions()

    best_iteration = 111
    inp = input("Please enter a question you want to ask: ")

    print("Searching questions with same meaning...")

    question = "Can you see who views your Instagram?"
    question = "How do i earn money online?"
    question = "Are we near World War 3?"
    question = str(inp)
    prepaire_traindata = True
    prepaire_traindata = False

    if prepaire_traindata:
        feature_extraction.extract_features(False, True)

    #script_evaluation.run_only([best_iteration])
    df_validation = pd.read_csv(data_paths.Get_Evaluation_Validation_Data_Path_For_Iteration(best_iteration), encoding="ISO-8859-1")

    found_duplicates = []
    for index, row in df_validation.iterrows():
        if row["question1"].lower() == question.lower() and row["predicted_binary"] == 1:
            found_duplicates.append(str(row["question2"]))# + str(row["is_duplicate"]))
        elif row["question2"].lower() == question.lower() and row["predicted_binary"] == 1:
            found_duplicates.append(str(row["question1"]))# + str(row["is_duplicate"]))

    if len(found_duplicates) == 0:
        print("Found no duplicate question.")
    else:
        print("Found some duplicate questions:")
        for item in sorted(set(found_duplicates))[:5]:
            print("- " + item)
