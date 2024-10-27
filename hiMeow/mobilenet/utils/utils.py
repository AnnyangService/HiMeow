import torch
import os
from hiMeow.mobilenet.model.trainMobilenet import trainMobilenet, train_model
from hiMeow.mobilenet.model.kFoldTrainMobilenet import k_fold_train
from hiMeow.mobilenet.utils.config import ProjectConfig


def save_model(model, disease):
    """ProjectConfig를 사용하여 모델 저장"""
    config = ProjectConfig()
    filepath = config.get_model_path(disease)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Model for {disease} saved to {filepath}")

def load_model(disease, num_classes=2, num_aux_features=3, device='cpu'):
    """ProjectConfig를 사용하여 모델 로드"""
    config = ProjectConfig()
    filepath = config.get_model_path(disease)
    if not os.path.exists(filepath):
        print(f"No existing model found for {disease}")
        return None
    model = trainMobilenet(num_classes=num_classes, num_aux_features=num_aux_features)
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    model.eval()
    return model

def use_loaded_model(model, input_data, aux_features):
    """모델 추론"""
    with torch.no_grad():
        output = model(input_data, aux_features)
    return output

def train_or_load_models(diseases, dataset, device, use_k_fold=False, k=5):
    """모델 학습 또는 로드"""
    config = ProjectConfig()
    disease_models = {}

    for disease in diseases:
        model_path = config.get_model_path(disease)

        if os.path.exists(model_path):
            print(f"Loading existing model for {disease}")
            model = load_model(disease, device=device)  # 수정된 load_model 사용
            if model is not None:
                disease_models[disease] = model
            else:
                print(f"Failed to load model for {disease}, training new model")
        else:
            print(f"Training new model for {disease}")
            model = trainMobilenet(num_classes=2, num_aux_features=3)
            model.to(device)

            if use_k_fold:
                model = k_fold_train(model, dataset, device, disease_name=disease, k=k)
            else:
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                model = train_model(model, dataloader, optimizer, criterion, device)
                save_model(model, disease)  # 수정된 save_model 사용

            disease_models[disease] = model

    return disease_models

# def test_models(disease_models, device):
#     for disease, model in disease_models.items():
#         sample_input = torch.randn(1, 3, 224, 224).to(device)
#         sample_aux_features = torch.randn(1, 3).to(device)
#         prediction = use_loaded_model(model, sample_input, sample_aux_features)
#         print(f"Sample prediction for {disease}:", prediction)