
from nltk.stem.wordnet import WordNetLemmatizer
from stanford_tagger import pos_tag
from nltk import bigrams
from nltk import ProjectiveDependencyParser

str1 = "What are the questions should not ask on ulf90ooi Quora?"

pdp = nltk.ProjectiveDependencyParser()
sent = str1.split()
trees = pdp.parse(sent)
for tree in trees:
    print(tree)

res1 = pos_tag(str1)

for word in res1:
    print(word)
    lmtzr = WordNetLemmatizer.lemmatize(word[1], word[0])
    print(lmtzr)

bigrams_val = bigrams(res1[0])

print(bigrams_val)


import nltk
mysentence = "a quick brown fox jumps over a lazy dog"
mysentencetokens_sw= nltk.word_tokenize(mysentence)
print(type(mysentencetokens_sw))

#Normalizing to lower
looper = 0
for token in mysentencetokens_sw:
        mysentencetokens_sw[looper] = token.lower()
        looper += 1
print("Normalized to lower -->")
print(mysentencetokens_sw)

#Removing stop words and small words
from nltk.corpus import stopwords
minlength = 2
mysentencetokens = [token for token in mysentencetokens_sw if (not token in stopwords.words('english')) and len(token) >= minlength]
print("Stop words removed -->")
print(mysentencetokens)

#Stemming
porter = nltk.PorterStemmer()
looper = 0
for token in mysentencetokens:
        mysentencetokens[looper] = porter.stem(token)
        looper += 1
print("Stemmed -->")
print(mysentencetokens)

#Lemmatization
lmtzr = nltk.stem.wordnet.WordNetLemmatizer()
looper = 0
for token in mysentencetokens:
    mysentencetokens[looper] = lmtzr.lemmatize(token)
    looper += 1
print("Lemmatized -->")
print(mysentencetokens)

#bigrams
bigrams_val = nltk.bigrams(mysentencetokens)

#trigrams
trigrams_val = nltk.trigrams(mysentencetokens)

#grams(text, n)

ngrams_val = nltk.ngrams(mysentencetokens, 7)

mytext = nltk.Text(mysentencetokens)

mytext.concordance("fox")

mytext.dispersion_plot(["fox", "dog", "cat", "rat"])

print(sorted(set(mytext)))

print(mytext.count("fox"))
#fdist1 = FreqDist(mytext)
#fdist1.plot(10, cumulative=False)