import settings

ROOT_DIR = ''

def init(root):
    global ROOT_DIR
    ROOT_DIR = root

def init_from_args(args):
    if len(args) > 1:
        init(args[1])

def Get_TEST_DATA_Path():
    return ROOT_DIR + 'test.csv'

def Get_TRAIN_DATA_Path():
    return ROOT_DIR + 'train.csv'

def Get_TEST_PREPROCESSED_Path():
    return ROOT_DIR + 'test_preprocessed.csv'

def Get_TRAIN_PREPROCESSED_Path():
    return ROOT_DIR + 'train_preprocessed.csv'

def Get_TEST_FEATURES_Path():
    return ROOT_DIR + 'features/test.csv'

def Get_TEST_FEATURES_BACKUP_Path():
    return ROOT_DIR + 'features/test_backup.csv'

def Get_IMPORTANT_INDICIES_Path():
    return ROOT_DIR + 'features/important_indicies.txt'

def Get_TRAIN_FEATURES_Path():
    return ROOT_DIR + 'features/train.csv'

def Get_TRAIN_FEATURES_BACKUP_Path():
    return ROOT_DIR + 'features/train_backup.csv'

def Get_Evaluation_Validation_Data_Path():
    return ROOT_DIR + 'evaluation/validation_data.csv'

def Get_Evaluation_Validation_Data_Path_For_Iteration(iteration):
    return ROOT_DIR + 'evaluation/runs/' + str(iteration) + '.csv'

def Get_MODEL_Path():
    return 'model.bin'

def Get_MODEL_DUMP_Path():
    return 'xgboost_model.pkl'

def Get_SUBMISSION_Path():
    return ROOT_DIR + 'submissions/prediction_xgb.csv'

def Get_FEATURES_PLT_Path():
    return ROOT_DIR + 'featuremap.png'

def Get_EVALUATION_PATH():
    return ROOT_DIR + 'evaluation/' + settings.run_name + '.csv'
