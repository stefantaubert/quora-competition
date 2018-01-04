import pandas as pd
import xgboost as xgb
from time import clock
import data_paths
import matplotlib.pyplot as plt

def predict_and_write_testdata():
    predict_and_write_data(data_paths.test, data_paths.test_features, data_paths.test_submission)

def predict_and_write_data(data, features, submission):

    # Zeit stoppen für das Vorhersagen der Test-Daten.
    start = clock()
    
    # Lade die zuvor berechneten Features für die Trainings-Daten.
    df_test = pd.read_csv(data, encoding="ISO-8859-1")
    x_test = pd.read_csv(features, encoding="ISO-8859-1")

    # Zuvor trainiertes Modell laden.
    bst = xgb.Booster(model_file=data_paths.model)
    
    # Test-Daten vorbereiten.
    d_test = xgb.DMatrix(x_test)

    # Test-Daten vorhersagen.
    p_test = bst.predict(d_test)

    #s_1 = 0.05
    # s_0 = 0.05
    #
    # p_test[p_test <= s_0] = 0
    # p_test[p_test >= 1 - s_1] = 1


    # Ergebnis speichern.
    sub = pd.DataFrame()
    sub['test_id'] = df_test['test_id']
    sub['is_duplicate'] = p_test
    sub.to_csv(submission, index=False)

    print('duration of prediction: ', round(clock()-start, 2), 's')
