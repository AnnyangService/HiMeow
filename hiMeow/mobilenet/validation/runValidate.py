from hiMeow.mobilenet.utils.utils import load_model
from hiMeow.mobilenet.utils.config import ProjectConfig
import torch
from tqdm import tqdm
import pandas as pd

from hiMeow.mobilenet.validation.loadTestDataset import load_validate_dataset
from hiMeow.mobilenet.validation.validator import ModelValidator


def run_validate(diseases=None, device=None, batch_size=16):
    config = ProjectConfig()

    if diseases is None:
        diseases = ['각막궤양', '결막염', '안검염', '백내장', '녹내장']

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, validate_loader = load_validate_dataset(batch_size=batch_size)
    print("Validate dataset loaded successfully")

    all_metrics = {}
    base_model = load_model(diseases[0], device=device)  # 첫 번째 모델로 validator 초기화
    validator = ModelValidator(
        model=base_model,
        device=device,
        results_dir=config.results_dir
    )

    for disease in diseases:
        print(f"\nValidate {disease} model...")

        model = load_model(disease, device=device)
        if model is None:
            continue

        # validator의 모델만 업데이트
        validator.model = model

        all_predictions = []
        all_labels = []

        for images, gender, age, eye_position, disease_nm, labels in tqdm(validate_loader):
            images = images.to(device)
            aux_features = torch.stack([gender, age, eye_position], dim=1).to(device)

            predicted = validator.validate_batch(images, aux_features)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

        metrics = validator.calculate_metrics(
            y_true=all_labels,
            y_pred=all_predictions,
            disease_name=disease
        )

        all_metrics[disease] = metrics

        print(f"\nResults for {disease} model:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")

    # 모든 질병 비교 그래프 생성
    validator.plot_all_diseases_comparison(all_metrics)

    # CSV 저장
    results_df = pd.DataFrame([
        {
            'Disease': disease,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1_Score': metrics['f1_score']
        }
        for disease, metrics in all_metrics.items()
    ])

    results_df.to_csv(f'{config.results_dir}/validate_results.csv', index=False)
    print(f"\nResults saved to {config.results_dir}")

    return all_metrics

if __name__ == '__main__':
    run_validate()