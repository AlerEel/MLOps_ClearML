import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from clearml import Task, Dataset
import seaborn as sns
from typing import List, Dict
import json

def get_experiment_metrics(task_id: str) -> Dict:
    """Получение метрик эксперимента по ID задачи"""
    task = Task.get_task(task_id=task_id)
    metrics = task.get_last_scalar_metrics()
    return metrics

def get_experiment_parameters(task_id: str) -> Dict:
    """Получение параметров эксперимента по ID задачи"""
    task = Task.get_task(task_id=task_id)
    parameters = task.get_parameters()
    return parameters

def plot_learning_curves(task_ids: List[str], save_path: str = None):
    """Визуализация кривых обучения для нескольких экспериментов"""
    plt.figure(figsize=(12, 6))
    
    for task_id in task_ids:
        task = Task.get_task(task_id=task_id)
        metrics = task.get_reported_scalars()
        
        if 'loss' in metrics:
            loss_values = [point['value'] for point in metrics['loss']]
            iterations = [point['iter'] for point in metrics['loss']]
            plt.plot(iterations, loss_values, label=f'Task {task_id}')
    
    plt.title('Learning Curves Comparison')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def compare_metrics(task_ids: List[str]) -> pd.DataFrame:
    """Сравнение финальных метрик экспериментов"""
    metrics_data = []
    
    for task_id in task_ids:
        metrics = get_experiment_metrics(task_id)
        parameters = get_experiment_parameters(task_id)
        
        metrics_data.append({
            'task_id': task_id,
            'final_loss': metrics.get('loss', None),
            'final_accuracy': metrics.get('accuracy', None),
            'embedding_dim': parameters.get('embedding_dim', None),
            'latent_dim': parameters.get('latent_dim', None),
            'batch_size': parameters.get('batch_size', None)
        })
    
    return pd.DataFrame(metrics_data)

def plot_parameter_impact(df: pd.DataFrame, parameter: str, metric: str, save_path: str = None):
    """Визуализация влияния параметра на метрику"""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=parameter, y=metric)
    plt.title(f'Impact of {parameter} on {metric}')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def generate_report(task_ids: List[str], output_dir: str = 'reports'):
    """Генерация полного отчета по экспериментам"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Сравнение метрик
    metrics_df = compare_metrics(task_ids)
    metrics_df.to_csv(os.path.join(output_dir, 'metrics_comparison.csv'), index=False)
    
    # Визуализация кривых обучения
    plot_learning_curves(task_ids, os.path.join(output_dir, 'learning_curves.png'))
    
    # Анализ влияния параметров
    for param in ['embedding_dim', 'latent_dim', 'batch_size']:
        plot_parameter_impact(metrics_df, param, 'final_loss', 
                            os.path.join(output_dir, f'{param}_impact.png'))
    
    # Сохранение детального отчета
    report = {
        'experiments': len(task_ids),
        'best_loss': metrics_df['final_loss'].min(),
        'best_accuracy': metrics_df['final_accuracy'].max(),
        'parameter_ranges': {
            'embedding_dim': {
                'min': metrics_df['embedding_dim'].min(),
                'max': metrics_df['embedding_dim'].max()
            },
            'latent_dim': {
                'min': metrics_df['latent_dim'].min(),
                'max': metrics_df['latent_dim'].max()
            },
            'batch_size': {
                'min': metrics_df['batch_size'].min(),
                'max': metrics_df['batch_size'].max()
            }
        }
    }
    
    with open(os.path.join(output_dir, 'experiment_report.json'), 'w') as f:
        json.dump(report, f, indent=4)

if __name__ == "__main__":
    # Пример использования
    task_ids = [
        "YOUR_TASK_ID_1",
        "YOUR_TASK_ID_2",
        "YOUR_TASK_ID_3"
    ]
    
    generate_report(task_ids) 