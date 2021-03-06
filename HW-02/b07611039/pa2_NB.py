#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import Libraries


# In[48]:


from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
import nltk
import pandas as pd
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
from sklearn.svm import SVC


# In[17]:


all_file = []

for i in range(1, 1096):
    ### read file
    doc_name = "~/NTUCourse/NTU-IM-ITM/HW-02/PA2-data/" + str(i) + ".txt"
    file = os.path.expanduser(doc_name)
    f = open(file)
    docs = f.read()
    #print(docs)
    
    ### lowerize
    token_lower = docs.lower()
    #print(token_lower)
    token = token_lower.replace('\n', '')    
    all_file.append([i, token])
    
docs = pd.DataFrame(all_file, columns = ['id', 'text'])
#print(docs['text'][0])
#print(docs)
#docs.head()


# In[18]:


print(len(docs))


# In[68]:


classes = [[11, 19, 29, 113, 115, 169, 278, 301, 316, 317, 321, 324, 325, 338, 341],
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16] ,
[813, 817, 818, 819, 820, 821, 822, 824, 825, 826, 828, 829, 830, 832, 833], 
[635, 680, 683, 702, 704, 705, 706, 708, 709, 719, 720, 722, 723, 724, 726], 
[646, 751, 781, 794, 798, 799, 801, 812, 815, 823, 831, 839, 840, 841, 842],
[995, 998, 999, 1003, 1005, 1006, 1007, 1009, 1011, 1012, 1013, 1014, 1015, 1016, 1019],
[700, 730, 731, 732, 733, 735, 740, 744, 752, 754, 755, 756, 757, 759, 760], 
[262, 296, 304, 308, 337, 397, 401, 443, 445, 450, 466, 480, 513, 533, 534], 
[130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 145], 
[31, 44, 70, 83, 86, 92, 100, 102, 305, 309, 315, 320, 326, 327, 328], 
[240, 241, 243, 244, 245, 248, 250, 254, 255, 256, 258, 260, 275, 279, 295], 
[535, 542, 571, 573, 574, 575, 576, 578, 581, 582, 583, 584, 585, 586, 588], 
[485, 520, 523, 526, 527, 529, 530, 531, 532, 536, 537, 538, 539, 540, 541]]

print(len(classes))
labels = []
for q in range(0, len(docs)):
    for i in range(0, 13):
        for j in range(0, 15):
            if q + 1 == classes[i][j]:
                #print(q, q+1, i, classes[i], i+ 1)
                labels.append([classes[i][j], i+1])
#print(labels)
labels = pd.DataFrame(sorted(labels, key = lambda l:l[0]), columns = ['training_id', 'classes'])
print(labels)        
training_docs = docs[docs['id'].isin(labels['training_id'])]
#print(training_docs)
testing_docs = docs[~docs['id'].isin(labels['training_id'])]
print(testing_docs['id'])


# In[37]:


bn_vec = CountVectorizer(binary = True)
brn_nb = bn_vec.fit_transform(docs)
train_text = []
train_label = []
#print(training_docs['id'][20])
#for i in range(195):
    #print(training_docs['id'][i])


# In[21]:


binary_vectorizer = CountVectorizer(binary = True)
binary_vectors = binary_vectorizer.fit_transform(training_docs['text'])
binary_vectors_test = binary_vectorizer.transform(testing_docs['text'])
x_train, x_test, y_train, y_test = train_test_split(binary_vectors, labels['classes'], test_size = 0.1)


# In[84]:


model = BernoulliNB()
model.fit(x_train, y_train)
prediction = []
expectation = []

prediction.extend(model.predict(x_test))
expectation.extend(y_test)

print("Precision, Recall, and F1 scores are as below.")
print(metrics.classification_report(expectation, prediction))
print("F1 scores =", metrics.f1_score(expectation, prediction, average='weighted'))
print("Precision =", metrics.precision_score(expectation, prediction, average='weighted'))
print("recall =", metrics.recall_score(expectation, prediction, average ='weighted'))


# In[88]:


precision = dict() 
recall = dict() 
for i in range(13):
    precision[i], recall[i], thresholds = metrics.precision_recall_curve(expectation, prediction, pos_label = (i + 1))
    plt.plot(recall[i], precision[i], lw = 2, label = 'class {}'.format(i + 1))

plt.xlabel("Recall") 
plt.ylabel("Precision")
plt.legend(loc = "upper right")
plt.title("Precision Recall Curve: Naïve Bayes Model")
plt.show()
#plt.savefig('P.png')
