import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from dataloader.dataLoader import CatEyeDatasetCustomized, data_transforms
from hiMeow.mobilenet.model.mobilenet import mobilenet, train_model
from hiMeow.mobilenet.model.utils import save_model, load_model, use_loaded_model

disease_models = {}

def train_or_load_models(diseases, dataloader, device):
    global disease_models
    for disease in diseases:
        model_path = f'models/mobilenet_v2_{disease}_model.pth'

        if os.path.exists(model_path):
            print(f"Loading existing model for {disease}")
            disease_models[disease] = load_model(disease, model_path, device=device)
        else:
            print(f"Training new model for {disease}")
            model = mobilenet(num_classes=2, num_aux_features=3)
            model.to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            trained_model = train_model(model, dataloader, optimizer, criterion, device)

            save_model(trained_model, disease, model_path)
            disease_models[disease] = trained_model


def test_models(device):
    for disease, model in disease_models.items():
        sample_input = torch.randn(1, 3, 224, 224).to(device)
        sample_aux_features = torch.randn(1, 3).to(device)
        prediction = use_loaded_model(model, sample_input, sample_aux_features)
        print(f"Sample prediction for {disease}:", prediction)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = CatEyeDatasetCustomized('../../dataset/Training', transform=data_transforms)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    print(f"Total samples: {len(dataset)}")

    diseases = ['각막궤양', '결막염', '안검염', '백내장', '녹내장']

    train_or_load_models(diseases, dataloader, device)
    print("Model loading/training complete for all diseases!")

    # test_models(device)


if __name__ == '__main__':
    main()