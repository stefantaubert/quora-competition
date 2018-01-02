'''Das Script trainiert das Model. Parameter: Root-Verzeichnis'''
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, recall_score, precision_score, f1_score
import xgboost as xgb
import data_paths
import settings

def plt_features(d_test):
    # Ausschlagskraft aller Features plotten
    bst = xgb.Booster(model_file=data_paths.model)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12,18))
    # print("Features names:")
    # print(d_test.feature_names)
    # print("Fscore Items:")
    # print(bst.get_fscore().items())
    mapper = {'f{0}'.format(i): v for i, v in enumerate(d_test.feature_names)}
    mapped = {mapper[k]: v for k, v in bst.get_fscore().items()}
    xgb.plot_importance(mapped, color='red', ax=ax)
    # plt.show()
    plt.draw()
    plt.savefig(data_paths.features_plt)


def train_and_save_model():
    # Zeit stoppen für das Trainieren des Modells.
    start = time.time()

    # Lade die zuvor berechneten Features für die Trainings-Daten.
    x_train = pd.read_csv(data_paths.train_features, encoding="ISO-8859-1")
    df_train = pd.read_csv(data_paths.train)

    # Ausgabedaten erstellen.
    y_train = df_train.is_duplicate

    # Trainings-Set und Validierungs-Set erstellen. Das Validierungs-Set enthält 10% aller Trainings-Daten.
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=settings.validation_size, random_state=settings.seed_validation_split_training)

    # Die Parameter für XGBoost erstellen.
    params = {}
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'logloss'
    params['eta'] = 0.02
    params['max_depth'] = settings.max_depth
    params['subsample'] = 0.6
    params['base_score'] = 0.2
    # params['scale_pos_weight'] = 0.36 #für test set

    # Berechnungen mit der GPU ausführen
    #params['updater'] = 'grow_gpu'

    # Datenmatrix für die Eingabedaten erstellen.
    d_train = xgb.DMatrix(x_train, label=y_train)
    d_valid = xgb.DMatrix(x_valid, label=y_valid)

    # Um den Score für das Validierungs-Set während des Trainings zu berechnen, muss eine Watchlist angelegt werden.
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    
    # Modell trainieren.
    # Geschwindigkeit ca. 1000 pro Minute auf der P6000
    # zeigt alle 10 Schritte den Score für das Validierungs-Set an
    print("Training model...")
    bst = xgb.train(params, d_train, settings.num_boosting_rounds, watchlist, early_stopping_rounds=50, verbose_eval=500)

    # Modell speichern.
    bst.dump_model(data_paths.model_dump)
    bst.save_model(data_paths.model)

    predicted = bst.predict(d_valid)

    predicted_values = predicted.copy()
    s = settings.rounding_boundary
    predicted_values[predicted_values < s] = 0
    predicted_values[predicted_values >= s] = 1

    print('calculated log_loss for validation-set: {}'.format(log_loss(y_valid, predicted)))
    print('calculated accuracy_score for validation-set: {}'.format(accuracy_score(y_valid, predicted_values)))
    print('calculated recall_score for validation-set: {}'.format(recall_score(y_valid, predicted_values)))
    print('calculated precision_score for validation-set: {}'.format(precision_score(y_valid, predicted_values)))
    print('calculated f1_score for validation-set: {}'.format(f1_score(y_valid, predicted_values)))

    print('model training duration: ', round(time.time()-start, 2) / 60, 'min')

