import nltk
from nltk.classify import apply_features

from nltk.corpus import names

labeled_names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])
import random

random.shuffle(labeled_names)

def gender_features(word):
    return {'suffix1': word[-1:], 'suffix2': word[-2:]}

def old():
   
    #featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
    train_set = apply_features(gender_features, labeled_names[500:])
    test_set = apply_features(gender_features, labeled_names[:500])

    #train_set, test_set = featuresets[500:], featuresets[:500]

    classifier = nltk.NaiveBayesClassifier.train(train_set)

    errors = []
    for (name, tag) in test_set:
        guess = classifier.classify(gender_features(name))
        if guess != tag:
            errors.append( (tag, guess, name) )

    print(errors)

    print(classifier.classify(gender_features('Neo')))
    print(classifier.classify(gender_features('Trinity')))
    print(nltk.classify.accuracy(classifier, test_set))

    classifier.show_most_informative_features(5)
    
train_names = labeled_names[1500:]
devtest_names = labeled_names[500:1500]
test_names = labeled_names[:500]
train_set = [(gender_features(n), gender) for (n, gender) in train_names]
devtest_set = [(gender_features(n), gender) for (n, gender) in devtest_names]
test_set = [(gender_features(n), gender) for (n, gender) in test_names]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, devtest_set))

errors = []
for (name, tag) in devtest_names:
    guess = classifier.classify(gender_features(name))
    if guess != tag:
        errors.append( (tag, guess, name) )

for (tag, guess, name) in sorted(errors):
    print('correct={:<8} guess={:<8s} name={:<30}'.format(tag, guess, name))