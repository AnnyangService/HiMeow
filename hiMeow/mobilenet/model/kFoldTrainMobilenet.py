import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from .trainMobilenet import train_model, trainMobilenet
from ..utils.config import ProjectConfig
import os


def k_fold_train(model, dataset, device, disease_name, k=5, num_epochs=5, batch_size=16):
    config = ProjectConfig()
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)

    best_accuracy = 0.0
    best_fold = 0

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold}')
        print('--------------------------------')

        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # 각 폴드별 모델 학습
        train_model(model, train_loader, optimizer, criterion, device, num_epochs)

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, gender, age, eye_position, _, labels in val_loader:
                inputs = inputs.to(device)
                aux_features = torch.stack([gender, age, eye_position], dim=1).float().to(device)
                labels = labels.to(device)

                outputs = model(inputs, aux_features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # 현재 폴드의 정확도 계산
        current_accuracy = 100 * correct / total
        print(f'Accuracy for fold {fold}: {current_accuracy}%')
        print('--------------------------------')

        # 최고 정확도 모델 저장
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_fold = fold
            model_path = os.path.join(config.models_dir, f'mobilenet_v2_{disease_name}_fold{fold}_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f'New best model saved: {model_path}')

    # 최종 결과 출력
    print(f'\nBest performing fold: {best_fold} with accuracy: {best_accuracy}%')

    # 최고 성능 모델 로드
    best_model_path = os.path.join(config.models_dir, f'mobilenet_v2_{disease_name}_fold{best_fold}_model.pth')
    model.load_state_dict(torch.load(best_model_path))

    return model