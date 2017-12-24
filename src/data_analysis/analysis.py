# import numpy  # linear algebra
# import pandas  # data processing, CSV file I/O (e.g. pd.read_csv)
# import nltk
# #from string_comparer import word_match_share
# import matplotlib.pyplot as plt
# import seaborn as sns
# from collections import Counter
#
# #get_ipython().magic('matplotlib inline')
# df_train = pandas.read_csv("../../data/train.csv")#, nrows=200)
# df_test = pandas.read_csv("../../data/test.csv")
# pal = sns.color_palette()
# all_questions = df_train['question1'].tolist() + df_train['question2'].tolist()
# #all_questions = numpy.unique(all_questions) mem error
# all_test_questions = df_test['question1'].tolist() + df_test['question2'].tolist()
#
# train_qs = pandas.Series(all_questions).astype(str)
# test_qs = pandas.Series(all_test_questions).astype(str)
# dist_train = train_qs.apply(lambda x: len(x.split()))
# #dist_test = test_qs.apply(lambda x: len(x.split()))
#
# def get_sentence_count(question):
#     sen = nltk.sent_tokenize(question)
#     return len(sen)
#
# #sen_c_train = train_qs.apply(get_sentence_count)
#
# #%%
# def get_sentence_count_histogram():
#     labels, values = zip(*Counter(sen_c_train).items())
#
#     for entry in range(0, len(labels)):
#         print("Sätze: {} Anzahl: {} Prozent: {}".format(labels[entry],values[entry], values[entry]/len(train_qs)*100))
#
# #get_sentence_count_histogram()
#
# # In[4]:
# def get_charcount_histogram():
#     dist_train = train_qs.apply(len)
#
#     dist_test = test_qs.apply(len)
#
#     plt.figure(figsize=(12.1, 7))
#     plt.hist(dist_train, bins=200, range=[0, 200], color='lightgrey', normed=True, label='Trainings-Set')
#     plt.hist(dist_test, bins=200, range=[0, 200], color='darkgrey', normed=True, alpha=0.5, label='Test-Set')
#     #plt.title('Normalisiertes Histogramm für die Anzahl der Zeichen pro Frage', fontsize=15)
#     plt.legend()
#     plt.xlabel('Anzahl der Zeichen', fontsize=15)
#     plt.ylabel('Auftrittswahrscheinlichkeit', fontsize=15)
#     plt.tight_layout(pad=0)
#     plt.draw()
#     plt.show()
#     plt.savefig("zeichenanzahl.pdf", format="pdf", bbox_inches='tight')
#     print('Durchschnitt {:.2f} Standardabweichung {:.2f} Maximum {:.2f} Minimum {:.2f}'.format(dist_train.mean(),
#                             dist_train.std(), dist_train.max() , dist_train.min()))
# get_charcount_histogram()
# # In[4]:
#
# def get_general_stats():
#     qids = pandas.Series(df_train['qid1'].tolist() + df_train['qid2'].tolist())
#
#     print('Anzahl an Fragepaaren im Trainings-Set: {}'.format(len(df_train)))
#     print('Anzahl an Fragepaaren im Test-Set: {}'.format(len(df_test)))
#     print('Anzahl an gleichen Fragepaaren: {}%'.format(round(df_train['is_duplicate'].mean() * 100, 2)))
#     print('Anzahl der Fragen im Traings-Set: {}'.format(len(numpy.unique(qids))))
#     print('Anzahl an Fragen die mehrmals auftreten: {}'.format(numpy.sum(qids.value_counts() > 1)))
# #get_general_stats()
#
# # In[5]:
# def get_wordcount_histogram():
#     plt.figure(figsize=(12.1, 7))
#     plt.hist(dist_train, bins=50, range=[0, 50], color=pal[2], normed=True, label='Trainings-Set')
#     plt.hist(dist_test, bins=50, range=[0, 50], color=pal[1], normed=True, alpha=0.5, label='Test-Set')
#     #plt.title('Normalisiertes Histogramm für die Anzahl an Wörtern pro Frage', fontsize=15)
#     plt.legend()
#     plt.xlabel('Anzahl an Wörtern', fontsize=15)
#     plt.ylabel('Auftrittswahrscheinlichkeit', fontsize=15)
#     plt.draw()
#     #plt.show()
#     plt.savefig("wortanzahl.pdf", format="pdf", bbox_inches='tight')
#     print('mean-train {:.2f} std-train {:.2f} mean-test {:.2f} std-test {:.2f} max-train {:.2f} max-test {:.2f}'.format(dist_train.mean(),
#                             dist_train.std(), dist_test.mean(), dist_test.std(), dist_train.max(), dist_test.max()))
#
# # In[6]:
# def get_wordcloud_train():
#     from wordcloud import WordCloud
#     cloud = WordCloud(width=2000, height=1000).generate(" ".join(train_qs.astype(str)))
#     plt.figure(figsize=(12, 6), facecolor='k')
#     plt.imshow(cloud)
#     plt.axis('off')
#     plt.tight_layout(pad=0)
#     plt.show()
#
# # In[6]:
# def get_wordcloud_test():
#     ####memory error
#     from wordcloud import WordCloud
#     cloud = WordCloud(width=2000, height=1000).generate(" ".join(test_qs.astype(str)))
#     plt.figure(figsize=(12, 6), facecolor='k')
#     plt.imshow(cloud)
#     plt.axis('off')
#     plt.tight_layout(pad=0)
#     plt.show()
#     plt.savefig("wordcloud", bbox_inches='tight')
#
# # In[6]:
# #Top 5 häufigsten Fragen im Trainigs-Set
# def get_top_5_questions_train():
#     qids = pandas.Series(df_train['qid1'].tolist() + df_train['qid2'].tolist())
#     qids_occurence_desc = qids.value_counts().index.tolist()
#
#     for i in range(5):
#         for index, row in df_train.iterrows():
#             if row['qid2'] == qids_occurence_desc[i]:
#                 print(row['question2'])
#                 break
#             elif row['qid1'] == qids_occurence_desc[i]:
#                 print(row['question1'])
#                 break
#
# #Fragen rausfischen die bestimmtes Muster erfüllen
# def get_matching_questions_simple(algo):
#     result = []
#     for index, row in df_train.iterrows():
#         frage1 = str(row['question1'])
#         frage2 = str(row['question2'])
#         if algo(frage1, frage2, row['is_duplicate']):
#             result.append((frage1, frage2))
#
#     print("Gefundene Treffer: {}".format(len(result)))
#     return result
#
# # In[39]:
#
# def get_question_equal_score(question1, question2):
#     q1_parts = question1.lower().split()
#     q2_parts = question2.lower().split()
#     equal_terms_count = 0
#     term_count = 0
#     if len(q1_parts) <= len(q2_parts):
#         term_count = len(q1_parts)
#         for str in q1_parts:
#             if str in q2_parts:
#                 equal_terms_count = equal_terms_count + 1
#     else:
#         term_count = len(q2_parts)
#         for str in q2_parts:
#             if str in q1_parts:
#                 equal_terms_count = equal_terms_count + 1
#
#     return [equal_terms_count, term_count]
#
# # In[27]:
#
# # einfache Fragen
# def compare_question_identical(question1, question2, is_duplicate):
#     if type(question1) is str and type(question2) is str:
#         scores = get_question_equal_score(question1, question2)
#         equal_terms_count = scores[0]
#         term_count = scores[1]
#         if is_duplicate == 0 and equal_terms_count == term_count and len(question1) == len(question2):
#             return True
#         else:
#             return False
#     else:
#         return False
#
#
# def get_all_stats():
#     get_general_stats()
#     #get_charcount_histogram()
#     #get_wordcount_histogram()
#     #get_wordcloud_train()
#     #get_wordcloud_test()
#     #get_top_5_questions_train()
#     pass
# #get_all_stats()
#
# def compare_word_match_share(q1, q2, dupl):
#     return word_match_share(q1, q2) > 0.9
#
# #print(get_matching_questions_simple(compare_question_identical))
