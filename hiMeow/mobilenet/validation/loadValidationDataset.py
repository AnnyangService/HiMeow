from hiMeow.mobilenet.dataloader.dataLoader import CatEyeDatasetCustomized, data_transforms
from hiMeow.mobilenet.utils.config import ProjectConfig
from torch.utils.data import DataLoader

def load_validation_dataset(batch_size=16, num_workers=0):
    """
    ProjectConfig를 사용하여 validation dataset을 로드합니다.

    Args:
        batch_size (int): 배치 크기 (기본값: 16)
        num_workers (int): 데이터 로딩에 사용할 워커 수 (기본값: 0)

    Returns:
        tuple: (validation_dataset, validation_loader)
            - validation_dataset: CatEyeDatasetCustomized 인스턴스
            - validation_loader: DataLoader 인스턴스
    """
    # ProjectConfig 인스턴스 가져오기
    config = ProjectConfig()

    # Validation 데이터셋 경로 확인
    validation_path = config.validation_path
    print(f"Loading validation dataset from: {validation_path}")

    # Validation 데이터셋 생성
    validation_dataset = CatEyeDatasetCustomized(
        data_dir=validation_path,
        transform=data_transforms
    )

    # 데이터셋 정보 출력
    print(f"Validation dataset size: {len(validation_dataset)}")

    # DataLoader 생성
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,  # validation은 셔플하지 않음
        num_workers=num_workers
    )

    return validation_dataset, validation_loader