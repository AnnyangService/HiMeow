# hiMeow/mobilenet/main.py
import torch
from hiMeow.mobilenet.utils.config import ProjectConfig
from hiMeow.mobilenet.dataloader.dataLoader import CatEyeDatasetCustomized, data_transforms
from hiMeow.mobilenet.utils.utils import train_or_load_models


def main():
    # 프로젝트 설정 초기화
    config = ProjectConfig()
    config.create_directories()

    # 디바이스 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 데이터셋 경로 설정 및 데이터 로드
    print(f"Dataset path: {config.train_path}")
    dataset = CatEyeDatasetCustomized(config.train_path, transform=data_transforms)
    print(f"Total samples: {len(dataset)}")

    diseases = ['각막궤양', '결막염', '안검염', '각막부골편', '비궤양성각막염']

    use_k_fold = input("Use k-fold cross validation? (y/n): ").lower() == 'y'
    k = 3 if use_k_fold else 1

    disease_models = train_or_load_models(diseases, dataset, device, use_k_fold=use_k_fold, k=k)
    print("Model loading/training complete for all diseases!")


if __name__ == '__main__':
    main()