import torch
import os
from hiMeow.mobilenet.model.trainMobilenet import trainMobilenet, train_model
from hiMeow.mobilenet.model.kFoldTrainMobilenet import k_fold_train

def save_model(model, disease, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Model for {disease} saved to {filepath}")

def load_model(disease, filepath, num_classes=2, num_aux_features=3, device='cpu'):
    if not os.path.exists(filepath):
        print(f"No existing model found for {disease}")
        return None
    model = trainMobilenet(num_classes=num_classes, num_aux_features=num_aux_features)
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    model.eval()
    return model

def use_loaded_model(model, input_data, aux_features):
    with torch.no_grad():
        output = model(input_data, aux_features)
    return output

def train_or_load_models(diseases, dataset, device, use_k_fold=False, k=5):
    disease_models = {}
    for disease in diseases:
        model_path = f'models/mobilenet_v2_{disease}_model.pth'

        if os.path.exists(model_path):
            print(f"Loading existing model for {disease}")
            disease_models[disease] = load_model(disease, model_path, device=device)
        else:
            print(f"Training new model for {disease}")
            model = trainMobilenet(num_classes=2, num_aux_features=3)
            model.to(device)

            if use_k_fold:
                model = k_fold_train(model, dataset, device, k=k)
            else:
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                model = train_model(model, dataloader, optimizer, criterion, device)

            save_model(model, disease, model_path)
            disease_models[disease] = model

    return disease_models

def test_models(disease_models, device):
    for disease, model in disease_models.items():
        sample_input = torch.randn(1, 3, 224, 224).to(device)
        sample_aux_features = torch.randn(1, 3).to(device)
        prediction = use_loaded_model(model, sample_input, sample_aux_features)
        print(f"Sample prediction for {disease}:", prediction)