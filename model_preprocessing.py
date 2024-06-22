import numpy as np
import pandas as pd
import pickle
from statistics import mode
import nltk
import datetime
from nltk import word_tokenize
from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords
from tensorflow.keras.models import Model
from tensorflow.keras import models
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input,LSTM,Embedding,Dense,Concatenate,Attention
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
start = datetime.datetime.now()
print('Время старта: ' + str(start))


df=pd.read_csv('train/train_1.csv')
input_data = df.loc[:,'Text']
target_data = df.loc[:,'Summary']
target_data.replace("", np.nan, inplace=True)

input_texts=[]
target_texts=[]
input_words=[]
target_words=[]
contractions=pickle.load(open("contractions.pkl","rb"))['contractions']
# initialize stop words and LancasterStemmer
stop_words=set(stopwords.words('english'))
stemm=LancasterStemmer()

def clean(texts,src):
  # remove the html tags
  texts = BeautifulSoup(texts, "lxml").text
  # tokenize the text into words
  words=word_tokenize(texts.lower())
  # filter words which contains \
  # integers or their length is less than or equal to 3
  words= list(filter(lambda w:(w.isalpha() and len(w)>=3),words))
  # contraction file to expand shortened words
  words = [contractions[w] if w in contractions else w for w in words]
  # stem the words to their root word and filter stop words
  if src == "inputs":
      words = [stemm.stem(w) for w in words if w not in stop_words]
  else:
      words = [w for w in words if w not in stop_words]
  return words

#pass the input records and taret records
for in_txt,tr_txt in zip(input_data,target_data):
  in_words= clean(in_txt,"inputs")
  input_texts+= [' '.join(in_words)]
  input_words+= in_words
  #add 'sos' at start and 'eos' at end of text
  tr_words= clean("sos "+tr_txt+" eos","target")
  target_texts+= [' '.join(tr_words)]
  target_words+= tr_words

# store only unique words from input and target list of words
input_words = sorted(list(set(input_words)))
target_words = sorted(list(set(target_words)))
num_in_words = len(input_words)  # total number of input words
num_tr_words = len(target_words)  # total number of target words

# get the length of the input and target texts which appears most often
max_in_len = mode([len(i) for i in input_texts])
max_tr_len = mode([len(i) for i in target_texts])

print("number of input words : ", num_in_words)
print("number of target words : ", num_tr_words)
print("maximum input length : ", max_in_len)
print("maximum target length : ", max_tr_len)

#split the input and target text into 80:20 ratio or testing size of 20%.
x_train,x_test,y_train,y_test=train_test_split(input_texts,target_texts,test_size=0.2,random_state=0)

# train the tokenizer with all the words
in_tokenizer = Tokenizer()
in_tokenizer.fit_on_texts(x_train)
tr_tokenizer = Tokenizer()
tr_tokenizer.fit_on_texts(y_train)

# convert text into sequence of integers
# where the integer will be the index of that word
x_train = in_tokenizer.texts_to_sequences(x_train)
y_train = tr_tokenizer.texts_to_sequences(y_train)

df_1 = pd.DataFrame({'x_train': x_train, 'y_train': y_train})
df_2 = pd.DataFrame({'x_test': x_test, 'y_test': y_test})

df_1.to_csv('train/prep_train_1.csv',index=False)
df_2.to_csv('train/prep_test_1.csv',index=False)

#фиксируем время окончания работы кода
finish = datetime.datetime.now()
# вычитаем время старта из времени окончания
print('Время работы: ' + str(finish - start))