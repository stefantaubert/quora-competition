#NLTK Book http://www.nltk.org
#Bird, Steven, Edward Loper and Ewan Klein (2009), Natural Language Processing with Python. O’Reilly Media Inc.

#Stanford Tagger: 
#https://nlp.stanford.edu/software/tagger.html
#http://www.nltk.org/api/nltk.tag.html#module-nltk.tag.stanford
#http://www.nltk.org/_modules/nltk/tokenize/stanford.html

#alle POS_TAGS: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

#Named Entity Recognizer (NER): https://nlp.stanford.edu/software/CRF-NER.html

 from nltk import RegexpParser
from nltk.tag import StanfordPOSTagger
from nltk.tokenize.stanford import StanfordTokenizer 
from nltk import word_tokenize
from os import environ
from nltk.tag import StanfordNERTagger
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Add the jar and model via their path (instead of setting environment variables):
pos_jar = 'C:/Portable/stanford-postagger-full-2017-06-09/stanford-postagger.jar'
pos_model = 'C:/Portable/stanford-postagger-full-2017-06-09/models/english-bidirectional-distsim.tagger'
ner_jar = "C:/Portable/stanford-ner-2017-06-09/stanford-ner.jar"
ner_model = "C:/Portable/stanford-ner-2017-06-09/classifiers/english.all.3class.distsim.crf.ser.gz"
java_path = "C:/Program Files/Java/jdk1.8.0_101/bin/java.exe"
environ['JAVAHOME'] = java_path

stanford_pos_tagger = StanfordPOSTagger(pos_model, pos_jar, encoding='utf8')
stanford_tokenizer = StanfordTokenizer(pos_jar, encoding='utf8')
stanford_ner_tagger = StanfordNERTagger(ner_model, ner_jar, encoding='utf-8')

print(stanford_tokenizer.tokenize("i'm we do an M.Phil in India after doing 3,45€ Masters in UK?"))


from nltk.parse.stanford import StanfordDependencyParser
path_to_jar = 'C:/Portable/stanford-parser-full-2017-06-09/stanford-parser.jar'
path_to_models_jar = 'C:/Portable/stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models.jar'
dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)

str1 = "Can I earn money on Quora?" 
str1 = "What's the PUK for TF64SIMC4?"
str1 = "How do I make friends."
str1 = "How did you get wealthy?"
str1 = "What is the most valuable thing you have?"
#str1 = "How to make friends ?"
def print_head_trees(str):
    result = dependency_parser.raw_parse(str1)
    print(result)
    for tree in result:
        tree.tree().draw()

#print_head_trees(str1)

def pos_tag(str):  
    return stanford_pos_tagger.tag(stanford_tokenizer.tokenize(str))

def ner_tag(str):  
    return stanford_ner_tagger.tag(stanford_tokenizer.tokenize(str))

def draw_pos_tree(str):
    tags = stanford_pos_tagger.tag(stanford_tokenizer.tokenize(str))
    pattern = """NP: {<DT>?<JJ>*<NN>}
    VBD: {<VBD>}
    IN: {<IN>}"""
    NPChunker = RegexpParser(pattern)
    result = NPChunker.parse(tags)
    result.draw() 

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

lemmatizer = WordNetLemmatizer()
def lemmatize(str):
    for tag in pos_tag(str):
        print(lemmatizer.lemmatize(tag[0], get_wordnet_pos(tag[1])))

ps = PorterStemmer()
def porter_stem(str):
    for w in stanford_tokenizer.tokenize(str):
        print(ps.stem(w))

stops = set(stopwords.words("english"))

def without_stops(str):
    for w in stanford_tokenizer.tokenize(str):
        if w not in stops:
            print(w)
#print(porter_stem(str1))   
print(without_stops(str1))

#print(pos_tag(str1))0
#print(ner_tag("Is USA the most powerful country of the world?"))
#print(ner_tag("Why is the USA the most powerful country of the world?"))