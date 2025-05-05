"""
Модуль для подготовки и обучения модели.
Создает и обучает нейронную сеть для генерации текстовых суммаризаций.
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
from .alternative_models import create_bidirectional_model, create_gru_model

# Получаем текущую задачу и закрываем её
main_task = Task.current_task()
main_task.close()

# Читаем ID датасета из файла
with open("config/task_ids.txt", "r") as f:
    task_ids = dict(line.strip().split("=") for line in f)
processed_dataset_id = task_ids["processed_dataset_id"]

# Загрузка необходимых ресурсов NLTK
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

# Загрузка обработанных данных
X_train = np.load('data/processed/X_train.npy')
X_val = np.load('data/processed/X_val.npy')
y_train = np.load('data/processed/y_train.npy')
y_val = np.load('data/processed/y_val.npy')

# Загрузка токенизатора
with open('data/processed/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Параметры модели
vocab_size = len(tokenizer.word_index) + 1
max_text_len = X_train.shape[1]
max_summary_len = y_train.shape[1]
embedding_dim = 256
latent_dim = 512

# Создание и обучение трех моделей
models_to_train = {
    "lstm": "Базовая модель с LSTM",
    "bidirectional": "Модель с двунаправленным LSTM",
    "gru": "Модель с GRU"
}

for model_name, description in models_to_train.items():
    # Создание задачи для каждой модели
    model_task = Task.init(
        project_name="Text Summarization Project",
        task_name=f"Model Training {model_name}",
        task_type=Task.TaskTypes.training
    )
    
    # Логирование параметров в ClearML
    model_task.connect({
        "vocab_size": vocab_size,
        "max_text_len": max_text_len,
        "max_summary_len": max_summary_len,
        "embedding_dim": embedding_dim,
        "latent_dim": latent_dim,
        "model_type": model_name,
        "model_description": description
    })
    
    # Создание модели
    if model_name == "lstm":
        # Базовая модель с LSTM
        encoder_inputs = Input(shape=(max_text_len,))
        encoder_embedding = Embedding(vocab_size, embedding_dim)(encoder_inputs)
        encoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None,))
        decoder_embedding = Embedding(vocab_size, embedding_dim)(decoder_inputs)
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

        attention = Attention()([decoder_outputs, encoder_outputs])
        decoder_concat = Concatenate(axis=-1)([decoder_outputs, attention])

        decoder_dense = Dense(vocab_size, activation='softmax')
        decoder_outputs = decoder_dense(decoder_concat)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    elif model_name == "bidirectional":
        model = create_bidirectional_model(vocab_size, max_text_len, max_summary_len, embedding_dim, latent_dim)
    else:  # gru
        model = create_gru_model(vocab_size, max_text_len, max_summary_len, embedding_dim, latent_dim)

    # Компиляция модели
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Подготовка данных для обучения
    decoder_input_data = y_train[:, :-1]
    decoder_target_data = y_train[:, 1:]
    decoder_input_val = y_val[:, :-1]
    decoder_target_val = y_val[:, 1:]

    decoder_target_data = decoder_target_data.reshape(decoder_target_data.shape[0], decoder_target_data.shape[1], 1)
    decoder_target_val = decoder_target_val.reshape(decoder_target_val.shape[0], decoder_target_val.shape[1], 1)

    # Обучение модели
    history = model.fit(
        [X_train, decoder_input_data],
        decoder_target_data,
        validation_data=([X_val, decoder_input_val], decoder_target_val),
        epochs=3,  # Уменьшено количество эпох
        batch_size=64
    )

    # Сохранение модели
    model.save(f'model_{model_name}.h5')

    # Логирование метрик в ClearML
    for epoch in range(len(history.history['loss'])):
        model_task.get_logger().report_scalar(
            "Loss",
            "Training",
            iteration=epoch,
            value=history.history['loss'][epoch]
        )
        model_task.get_logger().report_scalar(
            "Loss",
            "Validation",
            iteration=epoch,
            value=history.history['val_loss'][epoch]
        )
        model_task.get_logger().report_scalar(
            "Accuracy",
            "Training",
            iteration=epoch,
            value=history.history['accuracy'][epoch]
        )
        model_task.get_logger().report_scalar(
            "Accuracy",
            "Validation",
            iteration=epoch,
            value=history.history['val_accuracy'][epoch]
        )

    # Закрываем задачу
    model_task.close()

print("Все модели успешно обучены и сохранены")

