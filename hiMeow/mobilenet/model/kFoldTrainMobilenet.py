import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from .trainMobilenet import train_model

def k_fold_train(model, dataset, device, k=5, num_epochs=5, batch_size=16):
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold}')
        print('--------------------------------')

        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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

        print(f'Accuracy for fold {fold}: {100 * correct / total}%')
        print('--------------------------------')

    return model