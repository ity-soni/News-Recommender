from tfidf import *
import sys


with open(sys.argv[1]) as fd:
    xmltext = fd.read()
    text = gettext(xmltext)
    words=tokenizer(text)
    c = Counter(words)
    com=c.most_common(10)
    for k,v in com:
        print(f'{k} {v}')
        