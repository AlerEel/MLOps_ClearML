"""
Модуль для создания и подготовки исходного набора данных.
Выполняет загрузку данных из CSV файла, их очистку и разделение на тренировочные наборы.
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

# Читаем ID датасета из файла
with open("config/task_ids.txt", "r") as f:
    task_ids = dict(line.strip().split("=") for line in f)
raw_dataset_id = task_ids["raw_dataset_id"]

# Загрузка необходимых ресурсов NLTK
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

# Загрузка исходного датасета
# Используем только первые 200000 строк для ускорения обработки
df = pd.read_csv("data/raw/Reviews.csv", nrows=200000)

# Логирование параметров в ClearML
task.connect({"dataset_size": len(df)})

# Очистка данных
# Удаление дубликатов по полю Text
df.drop_duplicates(subset=['Text'], inplace=True)
# Удаление строк с пропущенными значениями
df.dropna(axis=0, inplace=True)

# Логирование статистики очистки
task.get_logger().report_table(
    "Data Cleaning Statistics",
    "Cleaning Results",
    table_plot=pd.DataFrame({
        "Original Size": [200000],
        "After Deduplication": [len(df)],
        "After NA Removal": [len(df)]
    })
)

# Выбор необходимых колонок
df = df.loc[:, ['Summary', 'Text']]

# Разделение данных на две части
df_1 = df.iloc[:10000, :]  # Первые 10000 записей
df_2 = df.iloc[10000:, :]  # Оставшиеся записи

# Сохранение разделенных наборов данных
df_1.to_csv('data/processed/train_1.csv', index=False)
df_2.to_csv('data/processed/train_2.csv', index=False)

# Получаем существующий датасет по ID
dataset = Dataset.get(dataset_id=raw_dataset_id)
dataset.add_files("data/processed/train_1.csv")
dataset.add_files("data/processed/train_2.csv")
dataset.upload()
dataset.finalize()

# Закрываем задачу
task.close()

print("Данные успешно подготовлены и загружены в ClearML")