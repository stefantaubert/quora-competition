import pandas  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy
from string_comparer import word_match_share

IS_DUPLICATE = "is_duplicate"
QUESTION1 = "question1"
QUESTION2 = "question2"
QID1 = "qid1"
QID2 = "qid2"
ID = "id"
WORDMATCH = "wordmatch"

df_train = pandas.read_csv("../data/train.csv")
df_train = df_train[df_train[IS_DUPLICATE] == 1]
df_train[WORDMATCH] = df_train.apply(lambda x: word_match_share(x[QUESTION1], x[QUESTION2]), axis =1)
df_train = df_train.sort_values([WORDMATCH], ascending=False)
df_train.ix[:,[WORDMATCH, QUESTION1,QUESTION2]].to_csv("../data/WordMatchesDuplicates.csv", sep='\t', encoding='utf-8')

# print_count = 5

# df_len = len(df_train)
# mod = round(df_len / print_count)

# def print_row(row):
#     print("ID: {}, WordMatch: {}\nQuestion 1: {}\nQuestion 2: {}".format(row[ID],row[WORDMATCH], row[QUESTION1],row[QUESTION2]))

# for index, row in df_train.iloc[[0, -1]].iterrows():
#     print_row(row)

# for index, row in df_train.iterrows():
#     if index % 10000 == 0:
#         print_row(row)