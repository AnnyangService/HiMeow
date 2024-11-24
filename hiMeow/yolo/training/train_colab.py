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

        # GPU 설정 (더 작은 배치 사이즈 사용)
        device = 'cuda:0' if check_gpu() else 'cpu'
        batch_size = 4  # 배치 사이즈를 4로 축소

        # 현재 작업 디렉토리 저장
        original_dir = Path.cwd()

        try:
            # wandb 비활성화
            os.environ['WANDB_DISABLED'] = 'true'

            # 데이터셋을 로컬로 복사
            if not local_disease_path.exists():
                print(f"Copying dataset to local storage...")
                source_path = dataset_base / disease_name
                # 복사할 데이터 크기 계산
                total_size = sum(f.stat().st_size for f in source_path.glob('**/*') if f.is_file())
                print(f"Total size to copy: {total_size / (1024 * 1024):.1f} MB")

                shutil.copytree(str(source_path), str(local_disease_path))
                print(f"Dataset copied to {local_disease_path}")

            # Python API를 통한 직접 학습
            print(f"\nStarting training for {disease_name} - {version}")
            print(f"Using device: {device}")
            print(f"Batch size: {batch_size}")
            print(f"Working directory: {local_disease_path}")

            # 작업 디렉토리 변경
            os.chdir(local_disease_path)

            # YOLO 모델 로드 및 학습 설정
            from ultralytics import YOLO
            import torch

            # CUDA 캐시 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 모델 로드
            model = YOLO(str(model_path))

            # 학습 실행
            results = model.train(
                data='datasets',
                epochs=100,
                imgsz=640,
                batch=batch_size,
                name=f'{disease_name}_{version}',
                device=device,
                exist_ok=True,
                amp=False,  # Mixed precision 비활성화
                workers=1,  # Worker 수 감소
                cache=False,  # 캐시 비활성화
                patience=50,
                save_period=10,
                pretrained=True,
                optimizer='SGD',
                lr0=0.01,
                weight_decay=0.0005,
                warmup_epochs=3,
                cos_lr=True,
                seed=42
            )

            # 모델 저장 처리
            source_model = Path('runs/classify/train/weights/best.pt')
            if source_model.exists():
                # 버전별 저장
                version_best = weights_dir / 'best.pt'
                shutil.copy2(str(source_model), str(version_best))
                print(f"Model saved to version directory: {version_best}")

                # 메인 디렉토리 저장
                main_best = base_output_dir / disease_name / 'best.pt'
                if not main_best.exists():
                    shutil.copy2(str(version_best), str(main_best))
                    print(f"Model copied to main directory: {main_best}")
                return True
            else:
                print(f"Warning: best.pt not found in {source_model}")
                return False

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
        # 원래 디렉토리로 복귀
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