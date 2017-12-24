
def word_count_features(sentence):
    return {'word_count' : len(sentence.split())}


print(word_count_features("Apples and oranges are similar."))
labeled_names()
featuresets = [(word_count_features(n), is_duplicate) for (q1,q2, is_duplicate) in labeled_names]

import nltk

print(nltk.help.upenn_tagset('RB'))

import spacy
import sys
is_64bits = sys.maxsize > 2**32

#nlp = spacy.load('en')


def word_vector():
    doc7 = nlp(u"Apples and oranges are similar. Boots and hippos aren't.")
    apples = doc7[0]
    oranges = doc7[2]
    boots = doc7[6]
    hippos = doc7[8]
    #geht iwie nicht
    print(apples.similarity(oranges))
    print(boots.similarity(hippos))

#word_vector()

def spacy_test():

    doc = nlp(u"I was suddenly logged off Gmail. I can't remember my Gmail password and just realized the recovery email is no longer alive. What can I do?")
    for sent in doc.sents:
        print(sent)

    print(doc)

#%%
def process_sentence(sentence):
    grammar = r"""
  NP: {<DT|PP\$>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and noun
      {<NNP>+}                # chunk sequences of proper nouns
"""
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(sentence)
    result.draw()
    return result

def ie_preprocess(document):
    sentences = nltk.sent_tokenize(document)
    print(sentences)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    print(sentences)
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    print(sentences)
    sentences = [process_sentence(sent) for sent in sentences]
    print(sentences)

ie_preprocess("Will I Why or why not?")