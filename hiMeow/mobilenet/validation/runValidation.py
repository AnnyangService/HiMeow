import torch
from hiMeow.mobilenet.utils.utils import load_model
from hiMeow.mobilenet.validation.loadValidationDataset import load_validation_dataset
from hiMeow.mobilenet.utils.config import ProjectConfig


def run_validation(diseases=None, device=None, batch_size=16):
    """
    여러 질병 모델에 대해 validation을 수행합니다.

    Args:
        diseases (list): 검증할 질병 이름 리스트 (None일 경우 기본 질병 리스트 사용)
        device (torch.device): 사용할 디바이스 (None일 경우 자동 선택)
        batch_size (int): 배치 크기
    """
    # 프로젝트 설정 초기화
    config = ProjectConfig()

    # 기본 질병 리스트 설정
    if diseases is None:
        diseases = ['각막궤양', '결막염', '안검염', '백내장', '녹내장']

    # 디바이스 설정
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 데이터셋 로드
    _, validation_loader = load_validation_dataset(batch_size=batch_size)
    print("Validation dataset loaded successfully")

    # 각 질병별로 모델 로드 및 validation 수행
    disease_models = {}
    for disease in diseases:
        print(f"\nProcessing disease: {disease}")

        # 모델 로드
        model = load_model(disease, device=device)
        if model is None:
            print(f"Failed to load model for {disease}")
            continue
        print(f"Model for {disease} loaded successfully")

        disease_models[disease] = model

    return disease_models


if __name__ == '__main__':
    disease_models = run_validation()
    print("\nValidation setup completed for all diseases!")