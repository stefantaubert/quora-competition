import settings

root_lines = open("../../config/data_root").read().split('\n')
root = root_lines[0]

if len(root_lines) >= 2:
    stanford_postagger = root_lines[1]

if len(root_lines) >= 3:
    java_home = root_lines[2]

test = root + '/test.csv'
train = root + '/train.csv'
test_preprocessed = root + '/test_preprocessed.csv'
train_preprocessed = root + '/train_preprocessed.csv'
test_features = root + '/features/test.csv'
test_features_backup = root + '/features/test_backup.csv'
train_features = root + '/features/train.csv'
train_features_backup = root + '/features/train_backup.csv'
validation = root + '/evaluation/validation_data.csv'
model = 'model.bin'
model_dump = 'xgboost_model.pkl'
submission = root + '/submissions/prediction_xgb.csv'
features_plt = root + 'featuremap.png'
evaluation = root + '/evaluation/' + settings.run_name + '.csv'

def Get_Evaluation_Validation_Data_Path_For_Iteration(iteration):
    return root + '/evaluation/runs/' + str(iteration) + '.csv'