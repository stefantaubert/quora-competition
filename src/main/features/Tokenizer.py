from nltk.tokenize.stanford import StanfordTokenizer
from os import environ
from nltk.tokenize import ToktokTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
import data_paths

nltk.data.path.append('/code/libs/nltk_data/')
#pos_jar = '/code/libs/stanford-postagger-full-2014-08-27/stanford-postagger.jar'
#pos_jar = '/code/libs/stanford-postagger-full-2017-06-09/stanford-postagger.jar'

stanford_tokenizer = StanfordTokenizer(data_paths.stanford_postagger, encoding='utf8')
#java_path = "/usr/lib/jvm/java-8-openjdk-amd64/bin/java"
environ['JAVAHOME'] = data_paths.java_home

class Tokenizer:
    def __init__(self):
        self.cache = {}
        self.tokenizer = ToktokTokenizer().tokenize
        self.lemmatizer = WordNetLemmatizer()


    def split(self, question):
        #return question.split()
        #return self.tokenizer(question)
        return list(map(self.lemmatizer.lemmatize, self.tokenizer(question)))
