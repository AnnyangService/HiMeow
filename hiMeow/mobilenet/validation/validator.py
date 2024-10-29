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
            _, predicted = torch.max(outputs, 1)
        return predicted

    def calculate_metrics(self, y_true, y_pred, disease_name):
        metrics = {}

        # numpy array로 변환하고 shape 확인
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # 1차원 array로 변환
        if len(y_true.shape) > 1:
            y_true = y_true.ravel()
        if len(y_pred.shape) > 1:
            y_pred = y_pred.ravel()

        # 메트릭 계산
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
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