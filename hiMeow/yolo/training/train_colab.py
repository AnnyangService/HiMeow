import os
import sys
import subprocess
import torch
import shutil
from pathlib import Path


def check_gpu():
    """GPU 사용 가능 여부 확인"""
    has_gpu = torch.cuda.is_available()
    if has_gpu:
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f}GB")
    else:
        print("No GPU available, using CPU")
    return has_gpu


def copy_results_to_drive(disease_name, version):
    """학습 결과를 Google Drive로 복사"""
    source_dir = Path('/content/temp_dataset') / disease_name / 'YOLOv8' / f'{disease_name}_{version}'
    dest_base = Path('/content/drive/MyDrive/hiMeow/yolo/training/models')
    dest_dir = dest_base / disease_name / version

    if source_dir.exists():
        print(f"Copying results for {disease_name} - {version}")

        # weights 디렉토리 복사
        weights_source = source_dir / 'weights'
        weights_dest = dest_dir / 'weights'
        weights_dest.mkdir(parents=True, exist_ok=True)

        if weights_source.exists():
            for pt_file in weights_source.glob('*.pt'):
                dest_file = weights_dest / pt_file.name
                shutil.copy2(str(pt_file), str(dest_file))
                print(f"Copied {pt_file.name}")

        # 기타 파일들 logs 디렉토리로 복사
        logs_dest = dest_dir / 'logs'
        logs_dest.mkdir(parents=True, exist_ok=True)

        for file_path in source_dir.glob('*'):
            if file_path.is_file():
                dest_path = logs_dest / file_path.name
                shutil.copy2(str(file_path), str(dest_path))
                print(f"Copied {file_path.name}")

        print("Copy completed!")
    else:
        print(f"Source directory not found: {source_dir}")


def train_model(disease_name, version):
    """단일 질병 모델 학습"""
    try:
        # 기본 경로 설정
        model_path = Path('/content/drive/MyDrive/hiMeow/yolo/training/yolov8m-cls.pt')
        dataset_base = Path('/content/drive/MyDrive/yolo_dataset/balanced')

        # 로컬 작업 디렉토리 설정
        local_base = Path('/content/temp_dataset')
        local_disease_path = local_base / disease_name

        # 모델 저장 경로 설정
        base_output_dir = Path('/content/drive/MyDrive/hiMeow/yolo/training/models')
        model_dir = base_output_dir / disease_name / version
        weights_dir = model_dir / 'weights'
        logs_dir = model_dir / 'logs'

        # 디렉토리 생성
        weights_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        local_base.mkdir(exist_ok=True)

        # GPU 설정
        device = 'cuda:0' if check_gpu() else 'cpu'
        batch_size = 4

        # 현재 작업 디렉토리 저장
        original_dir = Path.cwd()

        try:
            # wandb 비활성화
            os.environ['WANDB_DISABLED'] = 'true'

            # 데이터셋을 로컬로 복사
            if not local_disease_path.exists():
                print(f"Copying dataset to local storage...")
                source_path = dataset_base / disease_name
                total_size = sum(f.stat().st_size for f in source_path.glob('**/*') if f.is_file())
                print(f"Total size to copy: {total_size / (1024 * 1024):.1f} MB")

                shutil.copytree(str(source_path), str(local_disease_path))
                print(f"Dataset copied to {local_disease_path}")

            print(f"\nStarting training for {disease_name} - {version}")
            print(f"Using device: {device}")
            print(f"Batch size: {batch_size}")
            print(f"Working directory: {local_disease_path}")

            os.chdir(local_disease_path)

            from ultralytics import YOLO
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 체크포인트 확인 - 같은 버전에서만 이어서 학습
            last_checkpoint = None
            checkpoint_dir = local_disease_path / 'YOLOv8' / f'{disease_name}_{version}' / 'weights'
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

            # 새로운 버전이면 처음부터 시작
            model = YOLO(str(model_path) if not last_checkpoint else last_checkpoint)

            # 학습 실행
            results = model.train(
                data='datasets',
                epochs=100,
                imgsz=640,
                batch=batch_size,
                project='YOLOv8',
                name=f'{disease_name}_{version}',
                device=device,
                exist_ok=True,
                amp=False,
                workers=1,
                cache=False,
                patience=50,
                save=True,
                save_period=10,
                pretrained=True if not last_checkpoint else False,  # 체크포인트에서 시작하면 pretrained 비활성화
                optimizer='SGD',
                lr0=0.01,
                weight_decay=0.0005,
                warmup_epochs=3,
                cos_lr=True,
                seed=42,
                resume=True if last_checkpoint else False  # 체크포인트 있으면 이어서 학습
            )

            # 결과물 Google Drive로 복사
            copy_results_to_drive(disease_name, version)

            return True

        except Exception as e:
            print(f"Training error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    except Exception as e:
        print(f"Setup error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        os.chdir(original_dir)


def main():
    """메인 학습 파이프라인"""
    if len(sys.argv) != 2:
        print("Usage: python train.py <version>")
        print("Example: python train.py v1")
        sys.exit(1)

    version = sys.argv[1]
    diseases = ['각막궤양', '결막염', '안검염', '각막부골편', '비궤양성각막염']

    print(f"Starting training pipeline - Version {version}")
    device_type = 'GPU' if check_gpu() else 'CPU'
    print(f"Using device: {device_type}")

    for disease in diseases:
        print(f"\n{'=' * 50}")
        print(f"Starting training for {disease} - {version}")
        print(f"{'=' * 50}")

        success = train_model(disease, version)

        if success:
            print(f"\n=== Successfully completed training for {disease} - {version} ===")
            print(f"Model saved in: /content/drive/MyDrive/hiMeow/yolo/training/models/{disease}/{version}/")
            print(f"├── weights/")
            print(f"│   └── best.pt")
            print(f"├── logs/")
            print(f"└── config.yaml")
        else:
            print(f"\n=== Failed training for {disease} - {version} ===")
            response = input(f"Continue with next disease? (y/n): ")
            if response.lower() != 'y':
                print("Stopping training pipeline")
                break

    print("\nTraining pipeline completed")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")