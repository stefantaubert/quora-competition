from collections import Counter
import  itertools
from nltk.tokenize import ToktokTokenizer
import pandas  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
tokenizer = ToktokTokenizer()

df_train = pandas.read_csv("/datasets/sttau/train.csv")
df_test = pandas.read_csv("/datasets/sttau/test.csv")
all_questions = df_train['question1'].tolist() + df_train['question2'].tolist()
all_test_questions = df_test['question1'].tolist() + df_test['question2'].tolist()
train_qs = pandas.Series(all_questions).astype(str)
test_qs = pandas.Series(all_test_questions).astype(str)
dist_train = train_qs.apply(lambda x: tokenizer.tokenize(x))
dist_test = test_qs.apply(lambda x: tokenizer.tokenize(x))

c = Counter(list(itertools.chain.from_iterable(dist_train)))
mostcommon = c.most_common()
print("Trainings-Set")
print(mostcommon[:10])
print(mostcommon[-10:])

c = Counter(list(itertools.chain.from_iterable(dist_test)))
mostcommon = c.most_common()
print("Test-Set")
print(mostcommon[:10])
print(mostcommon[-10:])


import editdistance

print(editdistance.eval("How I can speak English fluently?", "How can I learn to speak English fluently?"))


from nltk.tokenize.stanford import StanfordTokenizer
from nltk.tokenize import ToktokTokenizer
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
pos_jar = '/code/libs/stanford-postagger-full-2014-08-27/stanford-postagger.jar'
stanford_tokenizer = StanfordTokenizer(pos_jar, encoding='utf8')

lmtzr = WordNetLemmatizer()
nltk.data.path.append('/code/libs/nltk_data/')
#nltk.download('wordnet')
q = "Are exocytosis and endocytosis examples of active or passive transport?"
print(ToktokTokenizer().tokenize(q))

print(WordNetLemmatizer().lemmatize("is","v"))
