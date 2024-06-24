import numpy as np
import pandas as pd
import pickle
from statistics import mode
import nltk
import torch
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

df = pd.read_csv('train/prep_train_1.csv')
input_texts = df['input_texts'].tolist()
target_texts = df['target_texts'].tolist()

num_in_words = 10610  # total number of input words
num_tr_words = 4292  # total number of target words
# get the length of the input and target texts which appears most often
max_in_len = 74
max_tr_len = 17

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

# pad array of 0's if the length is less than the maximum length
en_in_data = pad_sequences(x_train, maxlen=max_in_len, padding='post')
dec_data = pad_sequences(y_train, maxlen=max_tr_len, padding='post')

# decoder input data will not include the last word
# i.e. 'eos' in decoder input data
dec_in_data = dec_data[:, :-1]
# decoder target data will be one time step ahead as it will not include
# the first word i.e 'sos'
dec_tr_data = dec_data.reshape(len(dec_data), max_tr_len, 1)[:, 1:]

K.clear_session()
latent_dim = 500

# create input object of total number of encoder words
en_inputs = Input(shape=(max_in_len,))
en_embedding = Embedding(num_in_words + 1, latent_dim)(en_inputs)

# create 3 stacked LSTM layer with the shape of hidden dimension for text summarizer using deep learning
# LSTM 1
en_lstm1 = LSTM(latent_dim, return_state=True, return_sequences=True)
en_outputs1, state_h1, state_c1 = en_lstm1(en_embedding)

# LSTM2
en_lstm2 = LSTM(latent_dim, return_state=True, return_sequences=True)
en_outputs2, state_h2, state_c2 = en_lstm2(en_outputs1)

# LSTM3
en_lstm3 = LSTM(latent_dim, return_sequences=True, return_state=True)
en_outputs3, state_h3, state_c3 = en_lstm3(en_outputs2)

# encoder states
en_states = [state_h3, state_c3]

# Decoder.
dec_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(num_tr_words + 1, latent_dim)
dec_embedding = dec_emb_layer(dec_inputs)

# initialize decoder's LSTM layer with the output states of encoder
dec_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
dec_outputs, *_ = dec_lstm(dec_embedding, initial_state=en_states)

# Attention layer
attention = Attention()
attn_out = attention([dec_outputs, en_outputs3])

# Concatenate the attention output with the decoder outputs
merge = Concatenate(axis=-1, name='concat_layer1')([dec_outputs, attn_out])

#Dense layer (output layer)
dec_dense = Dense(num_tr_words+1, activation='softmax')
dec_outputs = dec_dense(merge)

#Model class and model summary for text Summarizer
model = Model([en_inputs, dec_inputs], dec_outputs)
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model.compile(
    optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(
    [en_in_data, dec_in_data],
    dec_tr_data,
    batch_size=512,
    epochs=10,
    validation_split=0.1,
)

# Save model
model.export("s2s")

