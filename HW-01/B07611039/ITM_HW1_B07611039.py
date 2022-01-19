#!/usr/bin/env python
# coding: utf-8

# In[9]:


# import packages
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import nltk


# [TODO]
# - pre-processor the data
#     - punc, lowercase, eng stop_words
# - vectorize the data
# - calculate the cosine similarity

# In[10]:


file = os.path.expanduser('~/NTUCourse/NTU-IM-ITM/HW-01/PA1-data/1094.txt')
f = open(file)

all_file = list(range(1, 1096))

for i in range(1, 1096):
    ### read file
    doc_name = "~/NTUCourse/NTU-IM-ITM/HW-01/PA1-data/" + str(i) + ".txt"
    file = os.path.expanduser(doc_name)
    f = open(file)
    docs = f.read()
    #print(docs)
    
    ### lowerize
    token_lower = docs.lower()
    #print(token_lower)
    
    ### stop punctuation
    token_sequence = word_tokenize(token_lower)
    stop_punc = [',',';','.','\'','?']
    tokens_wo_stop_punc = [x for x in token_sequence if x not in stop_punc]
    #print(tokens_wo_stop_punc)
    
    ### stop words in english
    stop_words = nltk.corpus.stopwords.words('english')
    tokens_wo_stop_words = [x for x in tokens_wo_stop_punc if x not in stop_words]
    #print(tokens_wo_stop_words)
    
    lst = ' '.join([str(elem) for elem in tokens_wo_stop_words])
    final_docs = []
    final_docs.append(lst)
    #print(final_docs)
    
    all_file[i-1] = lst
    
#print(all_file)
TFIDF_vectorizer = TfidfVectorizer()
TFIDF_vectors = TFIDF_vectorizer.fit_transform(all_file)
#print(TFIDF_vectors.toarray())
#print('\n')


# In[8]:


### outfile 
for i in range(1,1096):
    filename = "doc" + str(i) + ".txt"
    outF = open(filename, "w")
    for line in TFIDF_vectors.toarray()[i - 1]:
        outF.write(str(i) + "," + str(line))
        outF.write("\n")
    outF.close()


# In[12]:


### print the cosine similarity
print("The cosine similarity of 1.vec and 2.vec is",cosine_similarity(TFIDF_vectors[0], TFIDF_vectors[1]).flatten()[0])

