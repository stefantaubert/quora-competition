import pandas  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import data_paths

df_train = pandas.read_csv(data_paths.train)
df_test = pandas.read_csv(data_paths.test)
all_questions = df_train['question1'].tolist() + df_train['question2'].tolist()
all_test_questions = df_test['question1'].tolist() + df_test['question2'].tolist()
train_qs = pandas.Series(all_questions).astype(str)
test_qs = pandas.Series(all_test_questions).astype(str)
dist_train = train_qs.apply(len)
dist_test = test_qs.apply(len)

#plt.figure(figsize=(8,2.5))
plt.figure(figsize=(12,8))
#plt.hist(dist_train, bins=200, range=[0, 200], color='black',normed=True, label='Trainings-Set')

plt.hist(dist_train, bins=200, range=[0, 200], color='royalblue', alpha=0.4, normed=True, label='Trainings-Set')
plt.hist(dist_test, bins=200, range=[0, 200], color='seagreen', normed=True, alpha=0.5, label='Test-Set')

#plt.title('Histogramm für die Anzahl der Zeichen pro Frage', fontsize=15)
plt.legend()
plt.xlabel('Anzahl der Zeichen pro Frage', fontsize=14)
plt.ylabel('Häufigkeit (relativ)', fontsize=14)    
plt.tight_layout(pad=0)
plt.draw()
plt.show()
#plt.savefig("charcount.png", bbox_inches='tight')
#print('Durchschnitt {:.2f} Standardabweichung {:.2f} Maximum {:.2f} Minimum {:.2f}'.format(dist_train.mean(), df_train.std(), dist_train.max() , dist_train.min()))
