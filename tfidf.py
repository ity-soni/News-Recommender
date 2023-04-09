import sys

import nltk
from nltk.stem.porter import *
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import xml.etree.ElementTree as ET
from collections import Counter
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import zipfile
import os

def gettext(xmltext) -> str:
    """
    Parse xmltext and return the text from <title> and <text> tags
    """
    xmltext = xmltext.encode('ascii', 'ignore') # ensure there are no weird char
    xmlroot=ET.fromstring(xmltext)
    title=xmlroot.find('title')
    s=title.text
    for a in xmlroot.iterfind('./text/*'):
        s+=' '+a.text
    return s


def tokenize(text) -> list:
    """
    Tokenize text and return a non-unique list of tokenized words
    found in the text. Normalize to lowercase, strip punctuation,
    remove stop words, drop words of length < 3, strip digits.
    """
    text = text.lower()
    text = re.sub('[' + string.punctuation + '0-9\\r\\t\\n]', ' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if len(w) > 2 and w not in ENGLISH_STOP_WORDS]  # ignore a, an, to, at, be, ...
    return tokens


def stemwords(words) -> list:
    """
    Given a list of tokens/words, return a new list with each word
    stemmed using a PorterStemmer.
    """
    ps=PorterStemmer()
    new_words=[]
    for w in words:
        new_words.append(ps.stem(w))
    return new_words
    

def tokenizer(text) -> list:
    return stemwords(tokenize(text))


def compute_tfidf(corpus:dict) -> TfidfVectorizer:
    """
    Create and return a TfidfVectorizer object after training it on
    the list of articles pulled from the corpus dictionary. Meaning,
    call fit() on the list of document strings, which figures out
    all the inverse document frequencies (IDF) for use later by
    the transform() function. The corpus argument is a dictionary
    mapping file name to xml text.
    """
    tfidf = TfidfVectorizer(input='content',
                        analyzer='word',
                        preprocessor=gettext,
                        tokenizer=tokenizer,
                        stop_words='english', # even more stop words
                        decode_error = 'ignore')
    tfidfVectorizer=tfidf.fit(corpus.values())
    return tfidfVectorizer

def summarize(tfidf:TfidfVectorizer, text:str, n:int):
    """
    Given a trained TfidfVectorizer object and some XML text, return
    up to n (word,score) pairs in a list. Discard any terms with
    scores < 0.09. Sort the (word,score) pairs by TFIDF score in reverse order.
    """
    tscore=tfidf.transform([text])
    t0=tscore.todense().tolist()
    index=tfidf.get_feature_names()

    final=tuple(zip(index,t0[0]))
    sort_final=sorted(final, key=lambda tup: (tup[1],tup[0]), reverse=True)
    for i in range(len(sort_final)):
        if sort_final[i][1]<0.09:
            if n<i:
                return sort_final[:n]
            else:
                return sort_final[:i]

def load_corpus(zipfilename:str) -> dict:
    """
    Given a zip file containing root directory reuters-vol1-disk1-subset
    and a bunch of *.xml files, read them from the zip file into
    a dictionary of (filename,xmltext) associations. Use namelist() from
    ZipFile object to get list of xml files in that zip file.
    Convert filename reuters-vol1-disk1-subset/foo.xml to foo.xml
    as the keys in the dictionary. The values in the dictionary are the
    raw XML text from the various files.
    """
    prefix='/'.join(zipfilename.split('/')[:-1])
    with zipfile.ZipFile(zipfilename) as f:
        xmlFileList=f.namelist()[1:]
    zipDict={}
    for fileName in xmlFileList:
        fileName=prefix+'/'+fileName
        with open(fileName) as fd:
            zipDict[fileName.split('/')[-1]]=fd.read()
    return zipDict
