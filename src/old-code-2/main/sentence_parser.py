import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

stopwords = stopwords.words('english')

#Taken from Su Nam Kim Paper...
grammar = r"""
    NBAR:
        {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns
        
    NP:
        {<NBAR>}
        {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
"""
chunker = nltk.RegexpParser(grammar)
lemmatizer = WordNetLemmatizer()
stemmer = nltk.PorterStemmer()

def ParseAsTree(sentence):
    toks = nltk.word_tokenize(sentence)
    postoks = nltk.tag.pos_tag(toks)
    tree = chunker.parse(postoks)
    return tree

def getNP_leaves(tree):
    """Finds NP (nounphrase) leaf nodes of a chunk tree."""
    for subtree in tree.subtrees(filter = lambda t: t.label() =='NP'):
        yield subtree.leaves()

def normalise(word):
    """Normalises words to lowercase and stems and lemmatizes it."""
    #word = word.lower()
    #word = stemmer.stem(word)
    word = lemmatizer.lemmatize(word)
    return word

def is_acceptable_word(word):
    """Checks conditions for acceptable word: length, stopword."""
    accepted = bool(2 <= len(word) <= 40)
        #and word.lower() not in stopwords)
    return accepted

def getNP_terms(tree):
    res = []
    for leaf in getNP_leaves(tree):
        term = [normalise(w) for w,t in leaf if is_acceptable_word(w)]
        for word in term:
            res.append(word)
    return res

q1 = "How do parents feel when their kid has depression?" 
q2 = "What is it like to have a child with depression?"
tree1 = ParseAsTree(q1)
tree2 = ParseAsTree(q2)
#print(getNP_terms(tree1))
#print(getNP_terms(tree2))

a3 = "Can a non-profit organization invest in a profitable company in India?"
tree3 = ParseAsTree(a3)
#print(getNP_terms(tree3))

print(lemmatizer.lemmatize("organization", "n"))
print(lemmatizer.lemmatize("POLICY", "n"))

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

print(stemmer.stem("ORGANIZATION"))