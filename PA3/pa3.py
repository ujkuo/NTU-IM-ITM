# -*- coding: utf-8 -*-
"""Extract_embeddings.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tgWj26RP_paQioM0NBF3yxJfFkPTbxg4
"""

import tensorflow
from keras_bert import extract_embeddings
from keras_bert import load_vocabulary
from keras_bert import Tokenizer

model_path = './'
dict_path = './vocab.txt'


bert_token_dict = load_vocabulary(dict_path)
bert_tokenizer = Tokenizer(bert_token_dict)

single_input_text = ['all work and no play.']

tokens = bert_tokenizer.tokenize(single_input_text[0])
indices, segments = bert_tokenizer.encode(single_input_text[0])
print(len(tokens))
print(indices)
print(segments)
print(tokens)

embeddings = extract_embeddings(model_path, single_input_text)
print(len(embeddings))
print(len(embeddings[0]))
print(len(embeddings[0][0]))
print(embeddings[0][0])

from keras_bert import POOL_NSP, POOL_MAX
embeddings = extract_embeddings(model_path, single_input_text, poolings=[POOL_NSP])
print(len(embeddings))
print(embeddings)

two_input_text = [('all work and no play.', 'this is an order.')]

two_inputs_embeddings = extract_embeddings(model_path, two_input_text)
print(len(two_inputs_embeddings[0]))
print(two_inputs_embeddings[0][0])

tokens = bert_tokenizer.tokenize(first='all work and no play.', second='this is an order.')
indices, segments = bert_tokenizer.encode(first='all work and no play.', second='this is an order.')
print(len(tokens))
print(indices)
print(segments)
print(tokens)

import os
import numpy as np
batch_size = 100
data = './PA3-data'
dataset = []
for i in range(1, 1096):
  doc_name = "./PA3-data/" + str(i) + ".txt"
  file = os.path.expanduser(doc_name)
  f = open(file)
  docs = f.read().splitlines()
  docs = ''.join(docs)
  dataset.append(docs)

print(dataset[0])

#!nvidia-smi

import pickle
CLS_embedding_set = []
for i in range(0, len(dataset), batch_size):
  print("This is the "+ str(i // batch_size + 1) + " round out of " + str(len(dataset) // batch_size + 1))
  embeddings = extract_embeddings(model_path, dataset[i:i + batch_size])
  CLS_embedding_set.extend([item[0] for item in embeddings])

CLS_embedding_set = np.array(CLS_embedding_set)
with open('cls_embeddings.pickle', 'wb') as f:
  pickle.dump(CLS_embedding_set, f)

from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print(CLS_embedding_set)
print(len(CLS_embedding_set))

import pandas as pd

all_file = []

for i in range(1, 1096):
    ### read file
    doc_name = "./PA3-data/" + str(i) + ".txt"
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


docs = pd.DataFrame(all_file, columns = ['id', 'text'])
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
training_embedding = CLS_embedding_set[docs['id'].isin(labels['training_id'])]
print("training mbedding:")
print(len(training_embedding), training_embedding)
#print(training_docs)
testing_docs = docs[~docs['id'].isin(labels['training_id'])]
testing_embedding = CLS_embedding_set[~docs['id'].isin(labels['training_id'])]
print(testing_docs)
#print(len(testing_embedding), testing_embedding)

print(len(testing_embedding), testing_embedding)

x_train, x_test, y_train, y_test = train_test_split(training_embedding, labels['classes'], test_size = 0.1)
SVC_Linear_model = SVC(kernel='linear', C = 1.0)
SVC_Linear_model.fit(x_train, y_train)

prediction = []
expectation = []

prediction.extend(SVC_Linear_model.predict(x_test))
expectation.extend(y_test)

print("Precision, Recall, and F1 scores are as below.")
print(metrics.classification_report(expectation, prediction))
print("F1 scores :", metrics.f1_score(expectation, prediction, average='weighted'))
print("Precision :", metrics.precision_score(expectation, prediction, average='weighted'))
print("recall :", metrics.recall_score(expectation, prediction, average ='weighted'))


# In[89]:


precision = dict() 
recall = dict() 
for i in range(13):
    precision[i], recall[i], thresholds = metrics.precision_recall_curve(expectation, prediction, pos_label = (i + 1))
    plt.plot(recall[i], precision[i], lw = 2, label = 'class {}'.format(i + 1))

plt.xlabel("Recall") 
plt.ylabel("Precision")
plt.legend(loc = "upper right")
plt.title("Precision Recall Curve: SVM Linear")
plt.show()

result = pd.DataFrame()
result['Id'] = np.array(testing_docs['id'])
result['Value'] = SVC_Linear_model.predict(testing_embedding)

pd.DataFrame(result).to_csv('Embedding_svmLinear.csv', index = False)


