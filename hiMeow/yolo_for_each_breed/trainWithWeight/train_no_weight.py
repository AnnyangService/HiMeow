from ultralytics import YOLO
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import yaml
import shutil
import os

# 질병 클래스 정의
DISEASES = [
    'Corneal_Ulcer_Positive', 'Corneal_Ulcer_Negative',
    'Corneal_Secquestrum_Positive', 'Corneal_Secquestrum_Negative',
    'Conjunctivitis_Positive', 'Conjunctivitis_Negative',
    'Non_Ulcerative_Keratitis_Positive', 'Non_Ulcerative_Keratitis_Negative',
    'Blepharitis_Positive', 'Blepharitis_Negative'
]

# 품종별 질병 취약도 가중치 정의, 추후 수정 필요
BREED_VULNERABILITY = {
    'korean_shorthair': {
        'Corneal_Ulcer': 1.2,
        'Corneal_Secquestrum': 1.3,
        'Conjunctivitis': 1.1,
        'Non_Ulcerative_Keratitis': 1.2,
        'Blepharitis': 1.1
    }
}

# 품종별 임계값 정의
BREED_THRESHOLDS = {
    'korean_shorthair': {
        'Corneal_Ulcer': 0.5,
        'Corneal_Secquestrum': 0.5,
        'Conjunctivitis': 0.5,
        'Non_Ulcerative_Keratitis': 0.5,
        'Blepharitis': 0.5
    }
}


def check_gpu():
    """GPU 사용 가능 여부 확인"""
    has_gpu = torch.cuda.is_available()
    if has_gpu:
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f}GB")
    else:
        print("No GPU available, using CPU")
    return has_gpu


def copy_dataset_to_local():
    """데이터셋을 로컬 스토리지로 복사"""
    source_path = Path('../../../KoreaShortHair/datasets')
    local_path = Path('/content/temp_dataset/datasets')

    if not local_path.exists():
        print("Copying dataset to local storage...")
        total_size = sum(f.stat().st_size for f in source_path.glob('**/*') if f.is_file())
        print(f"Total size to copy: {total_size / (1024 * 1024):.1f} MB")

        shutil.copytree(str(source_path), str(local_path))
        print(f"Dataset copied to {local_path}")

    return local_path


def create_dataset_yaml(dataset_path):
    """데이터셋 YAML 파일 생성"""
    data_yaml = {
        'train': str(dataset_path / 'train'),
        'val': str(dataset_path / 'val'),
        'test': str(dataset_path / 'test'),
        'nc': len(DISEASES),
        'names': DISEASES
    }

    yaml_path = dataset_path / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)

    return str(yaml_path)


def check_checkpoint(save_dir, version):
    """체크포인트 확인"""
    checkpoint_dir = Path(save_dir) / f'yolo_cls_{version}' / 'weights'
    last_checkpoint = None

    if checkpoint_dir.exists():
        if (checkpoint_dir / 'last.pt').exists():
            last_checkpoint = str(checkpoint_dir / 'last.pt')
            print(f"Found checkpoint for {version}: {last_checkpoint}")
            print("Continuing training from last checkpoint...")
        else:
            checkpoints = list(checkpoint_dir.glob('epoch*.pt'))
            if checkpoints:
                last_checkpoint = str(max(checkpoints, key=lambda x: int(x.stem.replace('epoch', ''))))
                print(f"Found checkpoint {last_checkpoint}")
                print("Continuing training from last epoch checkpoint...")

    return last_checkpoint


def train_model(data_yaml, breed='korean_shorthair', version='v1', save_dir='runs/classify'):
    """모델 학습"""
    try:
        # wandb 비활성화
        os.environ['WANDB_DISABLED'] = 'true'

        # GPU 설정
        device = 'cuda:0' if check_gpu() else 'cpu'

        # 데이터셋 로컬로 복사
        local_dataset = copy_dataset_to_local()
        local_yaml = create_dataset_yaml(local_dataset)

        # 체크포인트 확인
        last_checkpoint = check_checkpoint(save_dir, version)

        # YOLO 모델 초기화
        base_model_path = 'yolov8m-cls.pt'
        model = YOLO(base_model_path if not last_checkpoint else last_checkpoint)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 학습 설정
        training_args = {
            'data': local_yaml,
            'epochs': 100,
            'batch': 16,
            'imgsz': 640,
            'device': device,
            'project': save_dir,
            'name': f'yolo_cls_{version}',
            'optimizer': 'SGD',
            'lr0': 0.01,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'cos_lr': True,
            'seed': 42,
            'workers': 8,
            'exist_ok': True,
            'pretrained': True if not last_checkpoint else False,
            'resume': True if last_checkpoint else False,
            'amp': False,
            'patience': 50,
            'save': True,
            'save_period': 10,  # 10 에포크마다 체크포인트 저장
            'cache': False
        }

        # 모델 학습
        results = model.train(**training_args)
        return model, results

    except Exception as e:
        print(f"Training error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def validate_model(model, data_yaml):
    """모델 검증"""
    val_args = {
        'data': data_yaml,
        'device': 0 if torch.cuda.is_available() else 'cpu',
        'verbose': True
    }

    results = model.val(**val_args)
    return results


def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: python train.py <version>")
        print("Example: python train.py v1")
        sys.exit(1)

    version = sys.argv[1]
    print(f"Starting training pipeline - Version {version}")

    # YAML 파일 생성
    data_yaml = create_dataset_yaml(Path('KoreaShortHair/datasets'))
    print(f"Dataset YAML created at: {data_yaml}")

    # 모델 학습
    model, train_results = train_model(data_yaml, version=version)

    if model is not None:
        print("Training completed!")

        # 모델 검증
        val_results = validate_model(model, data_yaml)
        print("Validation completed!")

    else:
        print("Training failed. Please check the error messages above.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        import traceback

        traceback.print_exc()