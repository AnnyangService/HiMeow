from hiMeow.mobilenet.dataloader.dataLoader import CatEyeDatasetCustomized, data_transforms
from hiMeow.mobilenet.utils.config import ProjectConfig
from torch.utils.data import DataLoader


def load_validate_dataset(batch_size=16, num_workers=0):
    # ProjectConfig 인스턴스 가져오기
    config = ProjectConfig()

    validate_path = config.validation_path
    print(f"Loading Test dataset from: {validate_path}")

    validate_dataset = CatEyeDatasetCustomized(
        data_dir=validate_path,
        transform=data_transforms
    )

    # 데이터셋 정보 출력
    print(f"Validate dataset size: {len(validate_dataset)}")

    # DataLoader 생성
    validate_loader = DataLoader(
        validate_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return validate_dataset, validate_loader
