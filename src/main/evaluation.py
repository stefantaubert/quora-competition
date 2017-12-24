import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, recall_score, precision_score, f1_score
import xgboost as xgb
import Paths
import matplotlib.pyplot as plt
import os
import json
import settings

i_0_30 = []
i_30_70 = []
i_70_150 = []
i_setted = False

def write_evaluation(root, current_run, save_validation_results):
    Paths.init(root)
    global i_0_30
    global i_30_70
    global i_70_150

    # Lade die zuvor berechneten Features für die Trainings-Daten.
    x_train = pd.read_csv(Paths.Get_TRAIN_FEATURES_Path(), encoding="ISO-8859-1")
    df_train = pd.read_csv(Paths.Get_TRAIN_DATA_Path())

    # Ausgabedaten erstellen.
    y_train = df_train.is_duplicate

    # Trainings-Set und Validierungs-Set erstellen. Das Validierungs-Set enthält 10% aller Trainings-Daten.
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=settings.validation_size, random_state=settings.seed_validation_split)

    write_initial_data(x_valid)

    write_scores(x_valid.ix[i_0_30], y_valid.ix[i_0_30], "0_30", current_run, False)
    write_scores(x_valid.ix[i_30_70], y_valid.ix[i_30_70], "30_70", current_run, False)
    write_scores(x_valid.ix[i_70_150], y_valid.ix[i_70_150], "70_150", current_run, False)
    write_scores(x_valid, y_valid, "all", current_run, save_validation_results)


def write_scores(x_valid, right_values, scope, run_id, save_validation_results):
    '''Der scope gibt an: all=alles,75=bis 75,75_150=75-150,150=ab 150.'''
    '''save_validation_results geht vorerst nur für all'''
    # Zuvor trainiertes Modell laden.
    bst = xgb.Booster(model_file=Paths.Get_MODEL_Path())

    # Test-Daten vorbereiten.
    d_valid = xgb.DMatrix(x_valid)

    # Test-Daten vorhersagen.
    orig_predicted_values = bst.predict(d_valid)
    data_normal_boundary = get_data(right_values, orig_predicted_values, settings.rounding_boundary, scope, run_id, x_valid)
    data_0_4 = get_data(right_values, orig_predicted_values, 0.4, scope, run_id, x_valid)
    data_0_6 = get_data(right_values, orig_predicted_values, 0.6, scope, run_id, x_valid)
    #
    # predicted_values = orig_predicted_values.copy()
    # s = settings.rounding_boundary
    # predicted_values[predicted_values < s] = 0
    # predicted_values[predicted_values >= s] = 1
    #
    # features = get_features(x_valid)
    # data = {
    #     "Iteration": run_id,
    #     "Zeitpunkt": time.strftime("%d-%m %H-%M-%S"),
    #     "Scope": scope,
    #     "Accuracy": accuracy_score(right_values, predicted_values),
    #     "Recall": recall_score(right_values, predicted_values),
    #     "Precision": precision_score(right_values, predicted_values),
    #     "F1": f1_score(right_values, predicted_values),
    #     "Log-Loss": log_loss(right_values, orig_predicted_values),
    #     #bei logloss kommt ein etwas anderer wert heraus als der direkt nach dem training,
    #     #da durch die zwischenspeicherung der daten ein paar kommastellen verloren gehen,
    #     #aber im angesicht der enorm verkürzten ausführungsdauer verschmerzbar ist.
    #     "Featureanzahl": len(features),
    #     "Features": json.dumps(features),
    # }

    if os.path.exists(Paths.Get_EVALUATION_PATH()):
        df = pd.read_csv(Paths.Get_EVALUATION_PATH(), encoding="ISO-8859-1")
        df = df.append(data_normal_boundary, ignore_index=True)
    else:
        df = pd.DataFrame(data=data_normal_boundary, index=[0], columns=data_normal_boundary.keys())

    df = df.append(data_0_4, ignore_index=True)
    df = df.append(data_0_6, ignore_index=True)

    df.to_csv(Paths.Get_EVALUATION_PATH(), index=False)

    if save_validation_results:
        predicted_values = orig_predicted_values.copy()
        s = settings.rounding_boundary
        predicted_values[predicted_values < s] = 0
        predicted_values[predicted_values >= s] = 1
        #Die ergebnisse für das Validierungsset speichern.
        df_validation = pd.read_csv(Paths.Get_Evaluation_Validation_Data_Path())
        #df_validation["right_values"] = right_values.tolist()
        df_validation["predicted_binary"] = predicted_values.tolist()
        df_validation["predicted"] = orig_predicted_values.tolist()
        df_validation.to_csv(Paths.Get_Evaluation_Validation_Data_Path_For_Iteration(run_id), index=False)

def get_data(right_values, orig_predicted_values, rounding_boundary, scope, run_id, x_valid):
    predicted_values = orig_predicted_values.copy()
    s = rounding_boundary
    predicted_values[predicted_values < s] = 0
    predicted_values[predicted_values >= s] = 1

    features = get_features(x_valid)
    data = {
        "Iteration": run_id,
        "Zeitpunkt": time.strftime("%d-%m %H-%M-%S"),
        "Scope": scope,
        "Rounding Boundary": rounding_boundary,
        "Accuracy": accuracy_score(right_values, predicted_values),
        "Recall": recall_score(right_values, predicted_values),
        "Precision": precision_score(right_values, predicted_values),
        "F1": f1_score(right_values, predicted_values),
        "Log-Loss": log_loss(right_values, orig_predicted_values),
        # bei logloss kommt ein etwas anderer wert heraus als der direkt nach dem training,
        # da durch die zwischenspeicherung der daten ein paar kommastellen verloren gehen,
        # aber im angesicht der enorm verkürzten ausführungsdauer verschmerzbar ist.
        "Featureanzahl": len(features),
        "Features": json.dumps(features),
    }

    return data

def write_initial_data(x_valid):
    '''Dauert ca 50 Sekunden, deswegen wird es nur 1x gemacht. Muss jedes mal gemacht wernde, wnn sich die Größe des Validierungs-Sets ändert.'''
    global i_setted

    if i_setted:
        return
    else:
        write_important_indicies(x_valid)
        write_selected_questions()
        i_setted = True


def write_important_indicies(x_valid):
    '''Dauert ca 50 Sekunden, deswegen wird es nur 1x gemacht. Muss jedes mal gemacht wernde, wnn sich die Größe des Validierungs-Sets ändert.'''
    global i_0_30
    global i_30_70
    global i_70_150

    i_0_30 = get_important_indicies(0, 30)
    i_30_70 = get_important_indicies(30, 70)
    i_70_150 = get_important_indicies(70, 150)

    print("Count of questions for all: " + str(len(x_valid)))
    print("Count of questions for 75: " + str(len(i_0_30)))
    print("Count of questions for 150_75: " + str(len(i_30_70)))
    print("Count of questions for 150: " + str(len(i_70_150)))

def write_selected_questions():
    x_train = pd.read_csv(Paths.Get_TRAIN_DATA_Path(), encoding="ISO-8859-1")
    y_train = x_train.is_duplicate
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=settings.validation_size, random_state=settings.seed_validation_split)
    x_valid.to_csv(Paths.Get_Evaluation_Validation_Data_Path(), index=False)

def get_important_indicies(min_len, max_len):
    x_train = pd.read_csv(Paths.Get_TRAIN_DATA_Path(), encoding="ISO-8859-1")

    # Ausgabedaten erstellen.
    y_train = x_train.is_duplicate

    # Trainings-Set und Validierungs-Set erstellen. Das Validierungs-Set enthält 10% aller Trainings-Daten.
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=settings.validation_size, random_state=settings.seed_validation_split)

    important_indices = []
    for index, row in x_valid.iterrows():
        # hier könnte man auch das len feature nehmen
        q1 = str(row["question1"])
        q2 = str(row["question2"])
        if (len(q1) <= max_len and len(q1) > min_len) and (len(q2) <= max_len and len(q2) > min_len):
            important_indices.append(index)

    return important_indices


def get_features(x_valid):
    # Zuvor trainiertes Modell laden.
    bst = xgb.Booster(model_file=Paths.Get_MODEL_Path())

    # Test-Daten vorbereiten.
    d_valid = xgb.DMatrix(x_valid)

    mapper = {'f{0}'.format(i): v for i, v in enumerate(d_valid.feature_names)}
    mapped = {mapper[k]: v for k, v in bst.get_fscore().items()}
    return {k: mapped[k] for k in sorted(mapped, key=mapped.get, reverse=True)}


def get_feature_importances(x_valid):
    # Zuvor trainiertes Modell laden.
    bst = xgb.Booster(model_file=Paths.Get_MODEL_Path())

    # Test-Daten vorbereiten.
    d_valid = xgb.DMatrix(x_valid)

    mapper = {'f{0}'.format(i): v for i, v in enumerate(d_valid.feature_names)}
    mapped = {mapper[k]: v for k, v in bst.get_fscore().items()}
    sorted_features = [(k, mapped[k]) for k in sorted(mapped, key=mapped.get, reverse=True)]
    res = ["Verwendete Features ({}):".format(len(sorted_features))]

    for feature, score in sorted_features:
        res.append("{}\t{}".format(score, feature))

    return res

def plt_features(bst, d_test):
    # Ausschlagskraft aller Features plotten
    fig, ax = plt.subplots(figsize=(12,18))
    mapper = {'f{0}'.format(i): v for i, v in enumerate(d_test.feature_names)}
    mapped = {mapper[k]: v for k, v in bst.get_fscore().items()}
    xgb.plot_importance(mapped, color='red', ax=ax)
    # plt.show()
    plt.draw()
    plt.savefig(Paths.Get_FEATURES_PLT_Path())
