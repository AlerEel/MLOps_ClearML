"""
Главный скрипт для запуска всего пайплайна обработки данных и обучения модели.
"""

import subprocess
import sys
import os
from clearml import Task
from .utils.timestamp import timestamp

def run_script(script_name, task_name=None):
    """Запускает Python скрипт и проверяет его выполнение"""
    print(f"\nЗапуск {script_name}...")
    
    # Если указано имя задачи, создаем новую задачу
    if task_name:
        task = Task.init(
            project_name="Text Summarization Project",
            task_name=task_name,
            task_type=Task.TaskTypes.training
        )
    
    try:
        # Запускаем скрипт как модуль с передачей временной метки
        env = os.environ.copy()
        env["CLEARML_TIMESTAMP"] = timestamp
        module_name = script_name.replace('/', '.').replace('.py', '')
        result = subprocess.run([sys.executable, '-m', module_name], capture_output=True, text=True, env=env)
        
        if result.returncode != 0:
            print(f"Ошибка при выполнении {script_name}:")
            print(result.stderr)
            if task_name:
                task.close()
            sys.exit(1)
        
        print(f"{script_name} успешно выполнен")
        print(result.stdout)
        
    except Exception as e:
        print(f"Ошибка при выполнении {script_name}: {str(e)}")
        if task_name:
            task.close()
        sys.exit(1)
    
    finally:
        if task_name:
            task.close()

def main():
    try:
        # Запускаем скрипты в правильном порядке
        run_script("src.utils.clearml_tasks", f"Pipeline Setup {timestamp}")
        run_script("src.data.data_creation", f"Data Creation {timestamp}")
        run_script("src.data.model_preprocessing", f"Model Preprocessing {timestamp}")
        run_script("src.models.model_preparation", f"Model Training {timestamp}")
        
        print("\nВесь пайплайн успешно выполнен!")
        
    except Exception as e:
        print(f"Ошибка при выполнении пайплайна: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 