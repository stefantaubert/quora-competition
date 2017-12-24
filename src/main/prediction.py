import pandas as pd
import xgboost as xgb
from time import clock
import Paths
import matplotlib.pyplot as plt

def predict_and_write_testdata(root):
    Paths.init(root)

    # Zeit stoppen für das Vorhersagen der Test-Daten.
    start = clock()
    
    # Lade die zuvor berechneten Features für die Trainings-Daten.
    df_test = pd.read_csv(Paths.Get_TEST_DATA_Path())
    x_test = pd.read_csv(Paths.Get_TEST_FEATURES_Path(), encoding="ISO-8859-1")

    # Zuvor trainiertes Modell laden.
    bst = xgb.Booster(model_file=Paths.Get_MODEL_Path())
    
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
    sub.to_csv(Paths.Get_SUBMISSION_Path(), index=False)

    print('duration of prediction: ', round(clock()-start, 2), 's')
