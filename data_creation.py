import numpy as np
import pandas as pd
import pickle
from statistics import mode
import nltk
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

#create the dataset1 file
df=pd.read_csv("Reviews.csv",nrows=200000)
df.drop_duplicates(subset=['Text'],inplace=True)
df.dropna(axis=0,inplace=True)
df = df.loc[:,['Summary', 'Text']]
df_1 = df.iloc[:100000,:]
df_2 = df.iloc[100000:,:]

df_1.to_csv('train/train_1.csv',index=False)
df_2.to_csv('train/train_2.csv',index=False)