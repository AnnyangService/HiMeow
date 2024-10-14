import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader.dataLoader import CatEyeDatasetCustomized, data_transforms
from hiMeow.mobilenet.model.mobilenet import mobilenet, train_model
from hiMeow.mobilenet.model.utils import load_model, use_loaded_model


def main():
    # GPU 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 데이터 로더 설정
    dataset = CatEyeDatasetCustomized('../../dataset/Training', transform=data_transforms)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    print(f"Total samples: {len(dataset)}")

    # 질병 리스트
    # diseases = ['각막궤양', '결막염', '안검염', '백내장', '녹내장']
    diseases = ['각막궤양']

    # 각 질병에 대한 모델 학습
    for disease in diseases:
        print(f"Training model for {disease}")

        # 모델 생성
        model = mobilenet(num_classes=2, num_aux_features=3)
        model.to(device)

        # 손실 함수 및 옵티마이저 정의
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # 모델 학습
        trained_model = train_model(model, dataloader, optimizer, criterion, device)

        # 모델 저장
        torch.save(trained_model.state_dict(), 'mobilenet_v2_with_aux_features_model.pth')

    print("Training complete for all diseases!")

    # # 모델 로딩 및 사용 예시
    # loaded_model = load_model(f'mobilenet_v2_{diseases[0]}_model.pth', device=device)
    #
    # # 예시 데이터로 모델 테스트
    # sample_input = torch.randn(1, 3, 224, 224).to(device)
    # sample_aux_features = torch.randn(1, 3).to(device)
    # prediction = use_loaded_model(loaded_model, sample_input, sample_aux_features)
    # print("Sample prediction:", prediction)

if __name__ == '__main__':
    main()
