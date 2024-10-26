import torch
import sys
import os

current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
project_root = os.path.dirname(os.path.dirname(current_dir))

if project_root not in sys.path:
    sys.path.append(project_root)

for path in sys.path:
    print(path)

from hiMeow.mobilenet.dataloader.dataLoader import CatEyeDatasetCustomized, data_transforms
from hiMeow.mobilenet.utils.utils import train_or_load_models


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 데이터셋 경로 설정
    dataset_path = os.path.join(project_root, 'dataset', 'Training')
    print(f"Dataset path: {dataset_path}")

    dataset = CatEyeDatasetCustomized(dataset_path, transform=data_transforms)
    print(f"Total samples: {len(dataset)}")

    diseases = ['각막궤양', '결막염', '안검염', '백내장', '녹내장']

    use_k_fold = input("Use k-fold cross validation? (y/n): ").lower() == 'y'
    k = 3 if use_k_fold else 1

    disease_models = train_or_load_models(diseases, dataset, device, use_k_fold=use_k_fold, k=k)
    print("Model loading/training complete for all diseases!")


if __name__ == '__main__':
    main()