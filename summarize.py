from tfidf import *

zipfilename = sys.argv[1]
summarizefile = sys.argv[2]

corp = load_corpus(zipfilename)
tfidf=compute_tfidf(corp)

sort_final=summarize(tfidf, corp[summarizefile], 20)

for i in sort_final:
    print(i[0],round(i[1],3))