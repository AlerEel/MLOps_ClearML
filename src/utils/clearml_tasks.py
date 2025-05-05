from clearml import Task, Dataset
from .timestamp import timestamp

# Создаем проект в ClearML
project_name = "Text Summarization Project"

# Задача 1: Подготовка данных
task_data_prep = Task.init(
    project_name=project_name,
    task_name=f"Data Preparation {timestamp}",
    task_type=Task.TaskTypes.data_processing
)
task_data_prep.set_comment("Подготовка и очистка исходных данных из Reviews.csv")

# Создаем датасет для сырых данных
dataset_raw = Dataset.create(
    dataset_name=f"Raw Reviews Dataset {task_data_prep.id}",
    dataset_project=project_name,
    dataset_tags=["raw", "reviews"]
)

# Задача 2: Предобработка данных
task_preprocessing = Task.create(
    project_name=project_name,
    task_name=f"Data Preprocessing {timestamp}",
    task_type=Task.TaskTypes.data_processing
)
task_preprocessing.set_comment("Предобработка текстовых данных (токенизация, стемминг, создание словарей)")

# Создаем датасет для обработанных данных
dataset_processed = Dataset.create(
    dataset_name=f"Processed Reviews Dataset {task_preprocessing.id}",
    dataset_project=project_name,
    dataset_tags=["processed", "reviews"]
)

# Задача 3: Обучение модели
task_model_training = Task.create(
    project_name=project_name,
    task_name=f"Model Training {timestamp}",
    task_type=Task.TaskTypes.training
)
task_model_training.set_comment("Обучение модели Seq2Seq с механизмом внимания")

# Устанавливаем зависимости между задачами
task_preprocessing.set_parent(task_data_prep.id)
task_model_training.set_parent(task_preprocessing.id)

# Сохраняем ID задач и датасетов в параметры задач
task_data_prep.connect({
    "raw_dataset_id": dataset_raw.id,
    "processed_dataset_id": dataset_processed.id,
    "preprocessing_task_id": task_preprocessing.id,
    "training_task_id": task_model_training.id
})

task_preprocessing.connect({
    "raw_dataset_id": dataset_raw.id,
    "processed_dataset_id": dataset_processed.id,
    "training_task_id": task_model_training.id
})

task_model_training.connect({
    "processed_dataset_id": dataset_processed.id
})

# Сохраняем ID задач в файл для использования в других скриптах
with open("config/task_ids.txt", "w") as f:
    f.write(f"data_prep_task_id={task_data_prep.id}\n")
    f.write(f"preprocessing_task_id={task_preprocessing.id}\n")
    f.write(f"training_task_id={task_model_training.id}\n")
    f.write(f"raw_dataset_id={dataset_raw.id}\n")
    f.write(f"processed_dataset_id={dataset_processed.id}\n")

print("Задачи и датасеты успешно созданы в ClearML") 