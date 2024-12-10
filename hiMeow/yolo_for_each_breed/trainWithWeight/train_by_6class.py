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
    'Blepharitis',
    'Conjunctivitis',
    'Corneal_Secquestrum',
    'Corneal_Ulcer',
    'Non_Ulcerative_Keratitis',
    'Normal'
]

BREED_VULNERABILITY = {
    'korean_shorthair': {
        'Corneal_Ulcer': 1.0,
        'Corneal_Secquestrum': 1.0,
        'Conjunctivitis': 1.0,
        'Non_Ulcerative_Keratitis': 1.0,
        'Blepharitis': 1.0
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
    import os

    current_file_dir = Path(__file__).parent  # 현재 스크립트 파일의 디렉토리
    source_path = current_file_dir / '../../..' / 'KoreaShortHair/datasets'
    source_path = source_path.resolve()  # 상대 경로를 절대 경로로 변환
    local_path = Path('/content/temp_dataset/datasets')

    if not local_path.exists():
        print("Copying dataset to local storage...")
        total_size = sum(f.stat().st_size for f in source_path.glob('**/*') if f.is_file())
        print(f"Total size to copy: {total_size / (1024 * 1024):.1f} MB")

        shutil.copytree(str(source_path), str(local_path))
        print(f"Dataset copied to {local_path}")

    return local_path


def check_checkpoint(save_dir, version):
    """체크포인트 확인"""
    checkpoint_dir = Path(
        '/content/drive/MyDrive/himeow/hiMeow/yolo_for_each_breed/trainWithWeight') / save_dir / f'yolo_cls_{version}' / 'weights'
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


def train_model(data_yaml, breed='korean_shorthair', version='v1', save_dir='results'):
    """모델 학습"""
    try:
        # wandb 비활성화
        os.environ['WANDB_DISABLED'] = 'true'
        os.environ['WANDB_MODE'] = 'offline'

        # GPU 설정
        device = 'cuda:0' if check_gpu() else 'cpu'

        # 데이터셋 로컬로 복사
        local_dataset = copy_dataset_to_local()

        # 체크포인트 확인
        last_checkpoint = check_checkpoint(save_dir, version)

        if last_checkpoint:
            print(f"Loading checkpoint: {last_checkpoint}")
            model = YOLO(last_checkpoint)  # 체크포인트가 있으면 사용
        else:
            print("No checkpoint found, starting from pretrained model")
            model = YOLO('yolov8m-cls.pt')

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 학습 설정
        training_args = {
            'data': str(local_dataset),
            'epochs': 100,
            'batch': 32,
            'imgsz': 640,
            'device': device,
            'project': str(save_dir),
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
            'amp': True,
            'patience': 50,
            'save': True,
            'save_period': 10,  # 10 에포크마다 체크포인트 저장
            'cache': True,
            'rect': True
        }

        # 모델 학습
        results = model.train(**training_args)
        return model, results

    except Exception as e:
        print(f"Training error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def validate_model(model, data_yaml, version='v1', save_dir='results'):
    """모델 검증"""
    val_args = {
        'data': data_yaml,
        'device': 0 if torch.cuda.is_available() else 'cpu',
        'verbose': True
    }

    results = model.val(**val_args)

    # validation 결과를 CSV 파일로 저장
    save_path = Path(save_dir) / f'yolo_cls_{version}' / 'final_validation_results.csv'
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 결과 저장
    if results is not None:
        # 기본 메트릭 저장
        metrics = {
            'top1_accuracy': results.top1,
            'top5_accuracy': results.top5,
        }
        with open(save_path, 'w') as f:
            f.write("Metric,Value\n")
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    f.write(f"{k},{v}\n")

        print(f"Validation results saved to {save_path}")

        # Confusion Matrix 생성 및 저장
        if hasattr(results, 'confusion_matrix'):
            confusion_matrix_path = save_path.parent / 'final_validation_confusion_matrix.csv'

            # 헤더 추가 (클래스 이름)
            header = ','.join(DISEASES)
            # confusion_matrix.matrix 속성 사용
            matrix = results.confusion_matrix.matrix
            if isinstance(matrix, np.ndarray):
                np.savetxt(confusion_matrix_path, matrix.astype(int),
                           delimiter=',', fmt='%d',
                           header=header)
                print(f"Confusion matrix saved to {confusion_matrix_path}")

    return results


def copy_results_to_drive(source_dir, version):
    """학습 결과를 Google Drive로 복사"""
    drive_path = Path('/content/drive/MyDrive/himeow/hiMeow/yolo_for_each_breed/trainWithWeight/results')
    source_dir = Path(source_dir) / f'yolo_cls_{version}'
    target_dir = drive_path / f'yolo_cls_{version}'

    if source_dir.exists():
        print(f"\nCopying results to Google Drive...")

        # 타겟 디렉토리 생성
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            # weights 폴더 복사
            if (source_dir / 'weights').exists():
                weights_target = target_dir / 'weights'
                weights_target.mkdir(exist_ok=True)
                for weight_file in (source_dir / 'weights').glob('*.pt'):
                    shutil.copy2(weight_file, weights_target / weight_file.name)
                print("Weights copied successfully")

            # 기타 결과 파일 복사
            for file in source_dir.glob('*.*'):  # weights 폴더 제외한 파일들
                if file.is_file():
                    shutil.copy2(file, target_dir / file.name)

            print("Results copied to Google Drive successfully")

        except Exception as e:
            print(f"Error copying results: {str(e)}")


def test_model(model, data_yaml, version='v1', save_dir='results'):
    """모델 테스트"""
    print("\nRunning final test...")
    test_args = {
        'data': data_yaml,
        'device': 0 if torch.cuda.is_available() else 'cpu',
        'verbose': True,
        'split': 'test'  # test 데이터셋 사용
    }

    results = model.val(**test_args)

    # test 결과를 CSV 파일로 저장
    save_path = Path(save_dir) / f'yolo_cls_{version}' / 'test_results.csv'
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if results is not None:
        # 기본 메트릭 저장
        metrics = {
            'top1_accuracy': results.top1,
            'top5_accuracy': results.top5,
        }
        with open(save_path, 'w') as f:
            f.write("Metric,Value\n")
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    f.write(f"{k},{v}\n")

        print(f"Test results saved to {save_path}")

        # Confusion Matrix 생성 및 저장
        if hasattr(results, 'confusion_matrix'):
            confusion_matrix_path = save_path.parent / 'test_confusion_matrix.csv'

            # 헤더 추가 (클래스 이름)
            header = ','.join(DISEASES)
            # confusion_matrix.matrix 속성 사용
            matrix = results.confusion_matrix.matrix
            if isinstance(matrix, np.ndarray):
                np.savetxt(confusion_matrix_path, matrix.astype(int),
                           delimiter=',', fmt='%d',
                           header=header)
                print(f"Confusion matrix saved to {confusion_matrix_path}")

    return results


def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: python train.py <version>")
        print("Example: python train.py v1")
        sys.exit(1)

    version = sys.argv[1]
    save_dir = 'results'
    checkpoint_dir = Path(
        '/content/drive/MyDrive/himeow/hiMeow/yolo_for_each_breed/trainWithWeight') / save_dir / f'yolo_cls_{version}'

    # validation, test 결과 파일 경로
    val_results_path = checkpoint_dir / 'final_validation_results.csv'
    test_results_path = checkpoint_dir / 'test_results.csv'

    if checkpoint_dir.exists() and (checkpoint_dir / 'weights/last.pt').exists():
        print(f"Found existing model for version {version}")

        # 기존 모델 로드
        model = YOLO(str(checkpoint_dir / 'weights/last.pt'))

        # validation 결과가 없는 경우에만 수행
        if not val_results_path.exists():
            print("Running validation...")
            local_dataset = Path('/content/temp_dataset/datasets')
            val_results = validate_model(model, str(local_dataset), version=version, save_dir=save_dir)
            print("Validation completed!")
            # Drive에 결과 복사
            copy_results_to_drive(save_dir, version)
        else:
            print("Validation results already exist.")

        # test 결과가 없는 경우에만 수행
        if not test_results_path.exists():
            print("Running test...")
            local_dataset = Path('/content/temp_dataset/datasets')
            test_results = test_model(model, str(local_dataset), version=version, save_dir=save_dir)
            print("Test completed!")
            # Drive에 결과 복사
            copy_results_to_drive(save_dir, version)
        else:
            print("Test results already exist.")

    else:
        print("No existing model found. Starting training...")
        # 모델 학습
        model, train_results = train_model(None, version=version)

        if model is not None:
            print("Training completed!")

            # validation과 test 수행
            local_dataset = Path('/content/temp_dataset/datasets')

            print("Running validation...")
            val_results = validate_model(model, str(local_dataset), version=version, save_dir=save_dir)
            print("Validation completed!")

            print("Running test...")
            test_results = test_model(model, str(local_dataset), version=version, save_dir=save_dir)
            print("Test completed!")

            # 모든 결과를 Drive에 복사
            copy_results_to_drive(save_dir, version)

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
