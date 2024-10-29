import os

from hiMeow.mobilenet.utils.utils import load_model
from hiMeow.mobilenet.utils.config import ProjectConfig
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from hiMeow.mobilenet.validation.loadTestDataset import load_validate_dataset


def run_validate(diseases=None, device=None, batch_size=16):
    config = ProjectConfig()

    if diseases is None:
        # diseases = ['각막궤양', '결막염', '안검염', '백내장', '녹내장']
        diseases = ['각막부골편', '비궤양성각막염']

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 결과 저장 디렉토리 설정
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 한글 폰트 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    _, validate_loader = load_validate_dataset(batch_size=batch_size)
    print("Validate dataset loaded successfully")

    # Loss 함수 정의
    criterion = torch.nn.BCEWithLogitsLoss()

    for disease in diseases:
        print(f"\nValidating {disease} model...")

        model = load_model(disease, device=device)
        if model is None:
            continue

        model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0

        with torch.no_grad():
            for images, gender, age, eye_position, disease_nm, _ in tqdm(validate_loader):
                images = images.to(device)
                aux_features = torch.stack([gender, age, eye_position], dim=1).to(device)

                # 현재 질병에 대한 라벨 생성
                labels = torch.zeros(len(disease_nm), 2, dtype=torch.float32).to(device)
                for i, d in enumerate(disease_nm):
                    labels[i][1 if d == disease else 0] = 1.0

                outputs = model(images, aux_features)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                # 예측값 계산
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend([1 if d == disease else 0 for d in disease_nm])

        # 평균 손실 계산
        avg_loss = total_loss / len(validate_loader)
        print(f"Average validation loss: {avg_loss:.4f}")

        # 성능 지표 계산
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)

        # 분류 보고서 생성
        report = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)

        # 결과 출력
        print(f"\nResults for {disease}:")
        print(f"Accuracy: {report['accuracy']:.4f}")
        print(f"Precision: {report['weighted avg']['precision']:.4f}")
        print(f"Recall: {report['weighted avg']['recall']:.4f}")
        print(f"F1 Score: {report['weighted avg']['f1-score']:.4f}")

        # 성능 지표 시각화
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        scores = [
            report['accuracy'],
            report['weighted avg']['precision'],
            report['weighted avg']['recall'],
            report['weighted avg']['f1-score']
        ]

        # 막대 그래프 생성
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, scores)
        plt.title(f'Performance Metrics - {disease}')
        plt.ylim(0, 1)

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.4f}', ha='center', va='bottom')

        plt.savefig(results_dir / f'metrics_bar_{disease}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 혼동 행렬 시각화
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {disease}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(results_dir / f'confusion_matrix_{disease}.png', dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\nResults saved to {results_dir}")


if __name__ == '__main__':
    run_validate()