import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ModelValidator:
    def __init__(self, model, device, results_dir):
        self.model = model
        self.device = device
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # 한글 폰트 설정
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False

    def validate_batch(self, images, aux_features):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(images, aux_features)
            _, predicted = torch.max(outputs.data, 1)
        return predicted

    def calculate_metrics(self, y_true, y_pred, disease_name):
        metrics = {}

        # 메트릭 계산
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary')
        metrics['recall'] = recall_score(y_true, y_pred, average='binary')
        metrics['f1_score'] = f1_score(y_true, y_pred, average='binary')
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

        # 1. Confusion Matrix 시각화
        plt.figure(figsize=(8, 6))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {disease_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(
            self.results_dir / f'confusion_matrix_{disease_name}.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

        # 2. 평가 지표 막대 그래프
        plt.figure(figsize=(10, 6))
        metrics_values = [metrics['accuracy'], metrics['precision'],
                          metrics['recall'], metrics['f1_score']]
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

        bars = plt.bar(metrics_names, metrics_values)
        plt.title(f'Performance Metrics - {disease_name}')
        plt.ylim(0, 1)  # 스케일을 0-1로 설정

        # 막대 위에 값 표시
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.4f}',
                     ha='center', va='bottom')

        plt.savefig(
            self.results_dir / f'metrics_bar_{disease_name}.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

        return metrics

    def plot_all_diseases_comparison(self, all_metrics):
        """
        모든 질병의 평가 지표를 한 그래프에 비교

        Args:
            all_metrics (dict): 질병별 메트릭 딕셔너리
        """
        diseases = list(all_metrics.keys())
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

        # 데이터 준비
        metrics_data = {
            metric: [all_metrics[disease][metric.lower()]
                     for disease in diseases]
            for metric in metrics_names
        }

        # 그래프 그리기
        plt.figure(figsize=(12, 6))
        x = np.arange(len(diseases))
        width = 0.2
        multiplier = 0

        for metric_name, metric_values in metrics_data.items():
            offset = width * multiplier
            plt.bar(x + offset, metric_values, width, label=metric_name)
            multiplier += 1

        # 그래프 꾸미기
        plt.xlabel('Diseases')
        plt.ylabel('Score')
        plt.title('Performance Metrics Comparison Across Diseases')
        plt.xticks(x + width * 1.5, diseases)
        plt.legend(loc='upper right')

        # 저장
        plt.savefig(
            self.results_dir / 'all_diseases_comparison.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()