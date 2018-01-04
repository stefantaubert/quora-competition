import pandas  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import data_paths

df_train = pandas.read_csv(data_paths.train)
df_test = pandas.read_csv(data_paths.test)

all_questions = df_train['question1'].tolist() + df_train['question2'].tolist()
all_test_questions = df_test['question1'].tolist() + df_test['question2'].tolist()
train_qs = pandas.Series(all_questions).astype(str)
test_qs = pandas.Series(all_test_questions).astype(str)
dist_train = train_qs.apply(lambda x: len(x.split()))
dist_test = test_qs.apply(lambda x: len(x.split()))

plt.figure(figsize=(12,8))
plt.hist(dist_train, bins=40, range=[0, 40], color='royalblue', alpha=0.4,normed=True, label='Trainings-Set')
plt.hist(dist_test, bins=40, range=[0, 40], color='seagreen', normed=True, alpha=0.5, label='Test-Set')
# plt.hist(dist_train, bins=40, range=[0, 40], color='black', alpha=0.6,normed=True, label='Trainings-Set')
# plt.hist(dist_test, bins=40, range=[0, 40], color='darkgrey', normed=True, alpha=0.7, label='Test-Set')
plt.legend()
plt.xlabel('Anzahl der Wörter pro Frage', fontsize=14)
plt.ylabel('Häufigkeit (relativ)', fontsize=14)    
plt.tight_layout(pad=0)
plt.draw()
plt.show()
#plt.savefig("wordcount_sw.png", format="png")
#print('Durchschnitt {:.2f} Standardabweichung {:.2f} Maximum {:.2f} Minimum {:.2f}'.format(dist_train.mean(), df_train.std(), dist_train.max() , dist_train.min()))
