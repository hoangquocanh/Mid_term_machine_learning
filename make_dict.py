import pandas as pd
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize

docs = pd.read_csv('tripadvisor_hotel_reviews.csv')
review = docs['Review']
stop_char = ['/', ';', '?', '+', '@', '_', '-', '*', 'Ã', '‡', '©']

for i in range(len(review)):
  for j in stop_char:
    review[i] = review[i].replace(j, ' ')
    review[i] = review[i].replace(',','.')

data = []
for word in review:
  for i in sent_tokenize(word):
    temp = []
    for j in word_tokenize(i):
      temp.append(j.lower())
    data.append(temp)

model = gensim.models.Word2Vec(data, min_count = 1, size = 120, window = 7)
model.save('dictionary.model')
print(model.similarity('nice', 'good'))
print(model.most_similar(positive='good', topn=15))
print(model.most_similar(positive='bad', topn=15))
