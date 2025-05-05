"""
Модуль для предварительной обработки данных перед обучением модели.
Выполняет токенизацию, создание словаря и подготовку последовательностей.
"""

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
from clearml import Task, Dataset
import os

# Получаем текущую задачу
task = Task.current_task()

# Читаем ID датасетов из файла
with open("config/task_ids.txt", "r") as f:
    task_ids = dict(line.strip().split("=") for line in f)
raw_dataset_id = task_ids["raw_dataset_id"]
processed_dataset_id = task_ids["processed_dataset_id"]

# Загрузка необходимых ресурсов NLTK
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

# Загрузка данных
df = pd.read_csv('data/processed/train_1.csv')

# Очистка текста
def clean_text(text):
    # Удаление HTML-тегов
    text = BeautifulSoup(text, "lxml").text
    # Токенизация
    tokens = word_tokenize(text.lower())
    # Удаление стоп-слов
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Стемминг
    stemmer = LancasterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(tokens)

# Применяем очистку к текстам и суммаризациям
df['Text'] = df['Text'].apply(clean_text)
df['Summary'] = df['Summary'].apply(clean_text)

# Создание токенизатора
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['Text'].tolist() + df['Summary'].tolist())

# Добавление специальных токенов
tokenizer.word_index['<start>'] = len(tokenizer.word_index) + 1
tokenizer.word_index['<end>'] = len(tokenizer.word_index) + 1
tokenizer.word_index['<pad>'] = 0

# Сохранение токенизатора
with open('data/processed/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Преобразование текстов в последовательности
text_sequences = tokenizer.texts_to_sequences(df['Text'].tolist())
summary_sequences = tokenizer.texts_to_sequences(df['Summary'].tolist())

# Добавление специальных токенов для начала и конца последовательности
summary_sequences = [[tokenizer.word_index['<start>']] + seq + [tokenizer.word_index['<end>']] for seq in summary_sequences]

# Паддинг последовательностей
max_text_len = max(len(seq) for seq in text_sequences)
max_summary_len = max(len(seq) for seq in summary_sequences)

text_sequences = pad_sequences(text_sequences, maxlen=max_text_len, padding='post', value=tokenizer.word_index['<pad>'])
summary_sequences = pad_sequences(summary_sequences, maxlen=max_summary_len, padding='post', value=tokenizer.word_index['<pad>'])

# Разделение на обучающую и валидационную выборки
X_train, X_val, y_train, y_val = train_test_split(text_sequences, summary_sequences, test_size=0.2, random_state=42)

# Сохранение обработанных данных
np.save('data/processed/X_train.npy', X_train)
np.save('data/processed/X_val.npy', X_val)
np.save('data/processed/y_train.npy', y_train)
np.save('data/processed/y_val.npy', y_val)

# Получаем существующий датасет по ID
dataset = Dataset.get(dataset_id=processed_dataset_id)
dataset.add_files("data/processed/X_train.npy")
dataset.add_files("data/processed/X_val.npy")
dataset.add_files("data/processed/y_train.npy")
dataset.add_files("data/processed/y_val.npy")
dataset.add_files("data/processed/tokenizer.pickle")
dataset.upload()
dataset.finalize()

# Закрываем задачу
task.close()

print("Данные успешно обработаны и загружены в ClearML")