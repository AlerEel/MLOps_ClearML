# Проект автоматического суммирования текстов с использованием нейронных сетей

## Описание проекта

Этот проект реализует систему автоматического суммирования текстов с использованием различных архитектур нейронных сетей:
1. Базовая модель с LSTM
2. Модель с двунаправленным LSTM
3. Модель с GRU

Проект интегрирован с платформой ClearML для отслеживания экспериментов, управления данными и мониторинга обучения моделей.

## Структура проекта

```
MLOps/
├── data/                      # Директория для данных
│   ├── raw/                  # Исходные данные
│   │   └── Reviews.csv       # Исходный датасет
│   └── processed/            # Обработанные данные
│       ├── train_1.csv      # Первая часть тренировочных данных
│       ├── train_2.csv      # Вторая часть тренировочных данных
│       ├── X_train.npy      # Обработанные входные данные
│       ├── X_val.npy        # Валидационные входные данные
│       ├── y_train.npy      # Обработанные целевые данные
│       └── y_val.npy        # Валидационные целевые данные
│
├── models/                   # Директория для моделей
│   ├── base/                # Базовая модель с LSTM
│   ├── bidirectional/       # Модель с двунаправленным LSTM
│   └── gru/                 # Модель с GRU
│
├── src/                     # Исходный код
│   ├── data/               # Скрипты для работы с данными
│   │   ├── data_creation.py
│   │   └── model_preprocessing.py
│   ├── models/             # Скрипты для моделей
│   │   ├── model_preparation.py
│   │   └── alternative_models.py
│   ├── utils/              # Вспомогательные скрипты
│   │   ├── clearml_tasks.py
│   │   └── timestamp.py
│   └── analysis/           # Скрипты для анализа
│       └── experiment_analysis.py
│
├── config/                  # Конфигурационные файлы
│   └── clearml.conf        # Конфигурация ClearML
│
├── reports/                 # Отчеты и результаты
│   ├── metrics/            # Метрики моделей
│   └── visualizations/     # Визуализации
│
├── notebooks/              # Jupyter ноутбуки
│   └── analysis.ipynb     # Анализ результатов
│
├── requirements.txt        # Зависимости проекта
├── run_pipeline.py        # Основной скрипт для запуска пайплайна
└── README.md              # Документация проекта
```

## Требования

Для работы проекта необходимы следующие библиотеки:
```
clearml>=1.12.0
tensorflow>=2.12.0
numpy>=1.21.0
pandas>=1.3.0
nltk>=3.6.0
scikit-learn>=0.24.0
beautifulsoup4>=4.9.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

## Установка и настройка

1. Создайте виртуальное окружение:
```bash
python -m venv .venv
source .venv/bin/activate  # для Linux/Mac
.venv\Scripts\activate     # для Windows
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. Настройте ClearML:
   - Создайте аккаунт на [ClearML](https://app.clear.ml)
   - Создайте файл `config/clearml.conf` с вашими учетными данными
   - Установите переменные окружения:
     ```bash
     export CLEARML_API_HOST="https://api.clear.ml"
     export CLEARML_WEB_HOST="https://app.clear.ml"
     export CLEARML_FILES_HOST="https://files.clear.ml"
     export CLEARML_API_ACCESS_KEY="YOUR_ACCESS_KEY"
     export CLEARML_API_SECRET_KEY="YOUR_SECRET_KEY"
     ```

## Пайплайн обработки данных

1. **Подготовка данных** (`src/data/data_creation.py`):
   - Загрузка данных из CSV файла
   - Очистка от дубликатов и пропущенных значений
   - Разделение на тренировочные наборы
   - Загрузка в ClearML

2. **Предобработка данных** (`src/data/model_preprocessing.py`):
   - Токенизация текста
   - Удаление стоп-слов
   - Стемминг
   - Создание последовательностей
   - Добавление специальных токенов
   - Паддинг последовательностей
   - Загрузка в ClearML

3. **Обучение моделей** (`src/models/model_preparation.py`):
   - Создание трех моделей:
     - Базовая модель с LSTM
     - Модель с двунаправленным LSTM
     - Модель с GRU
   - Обучение моделей
   - Сохранение моделей
   - Логирование метрик в ClearML

## Архитектуры моделей

### 1. Базовая модель с LSTM
- **Энкодер**: LSTM слой для обработки входного текста
- **Декодер**: LSTM слой для генерации суммаризации
- **Механизм внимания**: Позволяет модели фокусироваться на важных частях входного текста
- **Выходной слой**: Dense слой с softmax активацией

### 2. Модель с двунаправленным LSTM
- **Энкодер**: Двунаправленный LSTM для обработки текста в обоих направлениях
- **Декодер**: LSTM с увеличенным размером скрытого состояния
- **Механизм внимания**: Улучшенный механизм внимания
- **Выходной слой**: Dense слой с softmax активацией

### 3. Модель с GRU
- **Энкодер**: GRU слой для более быстрой обработки
- **Декодер**: GRU слой для генерации суммаризации
- **Механизм внимания**: Стандартный механизм внимания
- **Выходной слой**: Dense слой с softmax активацией

## Параметры моделей

- Размер словаря: определяется на основе входных данных
- Размер эмбеддингов: 256
- Размер скрытого состояния:
  - LSTM: 512
  - Bidirectional LSTM: 512 (1024 после конкатенации)
  - GRU: 512
- Оптимизатор: Adam
- Функция потерь: Sparse Categorical Crossentropy
- Метрика: Accuracy
- Количество эпох: 3
- Размер батча: 64

## Мониторинг в ClearML

В ClearML доступны:
- Графики обучения (loss, accuracy) для каждой модели
- Параметры моделей
- Статистика данных
- Визуализация архитектур моделей
- История экспериментов
- Сравнение моделей

## Анализ результатов

Для анализа результатов используйте:
1. `src/analysis/experiment_analysis.py` - автоматический анализ
2. `notebooks/analysis.ipynb` - интерактивный анализ

## Запуск проекта

1. Запустите основной скрипт:
```bash
python src/run_pipeline.py
```

Скрипт выполнит следующие действия:
- Создаст задачи и датасеты в ClearML
- Подготовит и очистит данные
- Выполнит предобработку текста
- Обучит все три модели
- Сохранит результаты в ClearML

## Лицензия

MIT License
