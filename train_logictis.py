
import gensim
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from nltk.tokenize import sent_tokenize, word_tokenize 
import numpy as np 
from sklearn.metrics import accuracy_score

print('--------Xử lý dữ liệu---------')
word_model = gensim.models.Word2Vec.load('dictionary.model')
Vocab = word_model.wv
list_vocab = list(Vocab.vocab)

docs = pd.read_csv('tripadvisor_hotel_reviews.csv')
Review =  docs['Review']
Rating = docs['Rating']

stop_char = [',', '/', ';', '?', '+', '@', '_', '-', '*', 'Ã', '‡', '©', '.']
stop_words = pd.read_table('stop_word.doc')
sw = stop_words['stop word']

for i in range(len(Review)):
  Review[i] = Review[i].lower()
  for j in stop_char:
    Review[i] = Review[i].replace(j, ' ')
  for j in sw:
    Review[i] = Review[i].replace(j, ' ')

def add_list(list_1, list_2):
  total = []
  for i in range(len(list_1)):
    total.append(list_1[i] + list_2[i])
  return total

Text = []
Label = []

for doc in Review:
  temp = []
  for word in word_tokenize(doc):
    if word not in list_vocab:
      continue
    if len(temp) == 0:
      temp = Vocab[word]
      continue
    temp2 = Vocab[word]
    temp = add_list(temp, temp2)
  for i in range(len(temp)):
    temp[i]=temp[i]/len(doc)
  Text.append(np.array(temp))

'''
wj = n_samples/(n_classes*n_samples(j))
n_samples = 20461
n_classes = 5

'''

for i in Rating:
  Label.append(i)
print('---------Training model---------')
X_train, X_test, Y_train, Y_test = train_test_split(Text, Label, test_size=0.2)

lg = LogisticRegression(solver = 'saga', random_state = 20, class_weight = {1:2.8798, 2:2.28232, 3:1.873718, 4:0.6776287, 5:0.451977})
lg.fit(X_train, Y_train)
pickle.dump(lg, open('hotel.model', 'wb'))

final_model = pickle.load(open('hotel.model', 'rb'))

result = final_model.predict(X_test)
print(result[:30])
print(Y_test[:30])
print(final_model.score(X_test, Y_test))
print('---------Done-----------')
