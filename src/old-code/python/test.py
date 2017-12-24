import numpy  # linear algebra
import pandas  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import csv
from sklearn.metrics import log_loss


# Create Submission File
def write_submission_file_for_test_data(values):
    df_test = pandas.read_csv('../data/test.csv')
    print(len(df_test))
    sub = pandas.DataFrame({'test_id': df_test['test_id'], 'is_duplicate': values})
    sub.to_csv('submission.csv', index=False)
    return

#ab = pandas.read_csv('../data/train.csv')
#b[ab['is_duplicate']==1].to_csv('abc.csv')

df_train = pandas.read_csv('../data/train.csv')#, nrows=200)


def compare_question_identical(question1, question2):
    if type(question1) is float or type(question2) is float :
        print("Float:" + question1)
        return

    if question1.lower() == question2.lower():
        return True
    else:
        return False


def get_matching_questions(algo):
    result = []
    for index, row in df_train.iterrows():
        frage1 = row['question1']
        frage2 = row['question2']
        if algo(frage1, frage2):
            result.append((frage1, frage2))
    return result


def simple_submission():
    mean = df_train['is_duplicate'].mean()  # Our predicted probability
    print(len(df_train))
    print('Predicted score:', log_loss(df_train['is_duplicate'], numpy.zeros_like(df_train['is_duplicate']) + mean))
    write_submission_file_for_test_data(df_train['is_duplicate'])
    return

matches = get_matching_questions(compare_question_identical)

if len(matches) > 0:
    print(len(matches))
    print(matches[0])
else:
    print("Keine Matches!")

print('Total number of question pairs for training: {}'.format(len(df_train)))
print('Duplicate pairs: {}%'.format(round(df_train['is_duplicate'].mean() * 100, 2)))

qids = pandas.Series(df_train['qid1'].tolist() + df_train['qid2'].tolist())
print('Total number of questions in the training data: {}'.format(len(numpy.unique(qids))))
print('Number of questions that appear multiple times: {}'.format(numpy.sum(qids.value_counts() > 1)))