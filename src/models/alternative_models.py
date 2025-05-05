import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, Attention, Bidirectional, GRU

def create_bidirectional_model(vocab_size, max_text_len, max_summary_len, embedding_dim=256, latent_dim=512):
    """Создание модели с двунаправленным LSTM"""
    # Энкодер
    encoder_inputs = Input(shape=(max_text_len,))
    encoder_embedding = Embedding(vocab_size, embedding_dim)(encoder_inputs)
    encoder_lstm = Bidirectional(LSTM(latent_dim, return_sequences=True, return_state=True))
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(encoder_embedding)
    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    encoder_states = [state_h, state_c]

    # Декодер
    decoder_inputs = Input(shape=(None,))
    decoder_embedding = Embedding(vocab_size, embedding_dim)(decoder_inputs)
    decoder_lstm = LSTM(latent_dim * 2, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

    # Механизм внимания
    attention = Attention()([decoder_outputs, encoder_outputs])
    decoder_concat = Concatenate(axis=-1)([decoder_outputs, attention])

    # Выходной слой
    decoder_dense = Dense(vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_concat)

    # Компиляция модели
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def create_gru_model(vocab_size, max_text_len, max_summary_len, embedding_dim=256, latent_dim=512):
    """Создание модели с GRU вместо LSTM"""
    # Энкодер
    encoder_inputs = Input(shape=(max_text_len,))
    encoder_embedding = Embedding(vocab_size, embedding_dim)(encoder_inputs)
    encoder_gru = GRU(latent_dim, return_sequences=True, return_state=True)
    encoder_outputs, state_h = encoder_gru(encoder_embedding)

    # Декодер
    decoder_inputs = Input(shape=(None,))
    decoder_embedding = Embedding(vocab_size, embedding_dim)(decoder_inputs)
    decoder_gru = GRU(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _ = decoder_gru(decoder_embedding, initial_state=state_h)

    # Механизм внимания
    attention = Attention()([decoder_outputs, encoder_outputs])
    decoder_concat = Concatenate(axis=-1)([decoder_outputs, attention])

    # Выходной слой
    decoder_dense = Dense(vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_concat)

    # Компиляция модели
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model 