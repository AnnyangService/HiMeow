import torch
from dataloader.dataLoader import CatEyeDatasetCustomized, data_transforms
from hiMeow.mobilenet.utils.utils import train_or_load_models


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = CatEyeDatasetCustomized('../../dataset/Training', transform=data_transforms)
    print(f"Total samples: {len(dataset)}")

    diseases = ['각막궤양', '결막염', '안검염', '백내장', '녹내장']

    use_k_fold = input("Use k-fold cross validation? (y/n): ").lower() == 'y'
    k = 3 if use_k_fold else 1

    disease_models = train_or_load_models(diseases, dataset, device, use_k_fold=use_k_fold, k=k)
    print("Model loading/training complete for all diseases!")

if __name__ == '__main__':
    main()