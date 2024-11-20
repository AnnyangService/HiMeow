import os
import sys
import subprocess
import torch
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
    # 기본 경로 설정
    model_path = Path('C:/Users/jjj53/Desktop/himeow/hiMeow/yolo/training/yolov11m-cls.pt')
    dataset_base = Path('C:/Users/jjj53/Desktop/himeow/yolo_dataset/balanced')

    # 작업 디렉토리를 disease 폴더로 변경
    disease_path = dataset_base / disease_name

    # 모델 저장 경로 설정
    model_dir = Path(f'models/{disease_name}/{version}')
    weights_dir = model_dir / 'weights'
    logs_dir = model_dir / 'logs'
    weights_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # GPU 설정
    device = 'cuda:0' if check_gpu() else 'cpu'
    batch_size = 32 if device == 'cuda:0' else 16

    # 현재 작업 디렉토리 저장
    original_dir = Path.cwd()

    # 명령어 리스트 설정
    cmd = [
        'yolo',
        'classify',
        'train',
        f'model={str(model_path)}',
        'data=datasets',  # 상대 경로 사용
        'epochs=100',
        'imgsz=640',
        f'batch={batch_size}',
        'optimizer=Adam',
        'lr0=0.001',
        f'name={disease_name}_{version}',
        f'project={str(logs_dir)}',
        f'device={device}',
        'verbose=True'
    ]

    try:
        # disease 폴더로 이동
        os.chdir(disease_path)

        print(f"\nStarting training for {disease_name} - {version}")
        print(f"Using device: {device}")
        print(f"Batch size: {batch_size}")
        print(f"Working directory: {disease_path}")
        print(f"Logs will be saved to: {logs_dir}")

        # 실시간 출력을 위한 Popen 사용
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )

        # 실시간으로 출력 처리
        while True:
            output = process.stdout.readline()
            error = process.stderr.readline()

            if output:
                print(output.strip())
            if error:
                print(error.strip())

            # 프로세스가 종료되었는지 확인
            if output == '' and error == '' and process.poll() is not None:
                break

        if process.returncode != 0:
            print(f"Training failed with return code: {process.returncode}")
            return False

        # 모델 저장 처리
        best_model = logs_dir / f'{disease_name}_{version}/weights/best.pt'
        if best_model.exists():
            version_best = weights_dir / 'best.pt'
            os.replace(str(best_model), str(version_best))
            print(f"Model saved to version directory: {version_best}")

            main_best = Path(f'models/{disease_name}/best.pt')
            if not main_best.exists():
                os.copy2(str(version_best), str(main_best))
                print(f"Model copied to main directory: {main_best}")
            return True
        else:
            print("Warning: best.pt not found after training")
            return False

    except Exception as e:
        print(f"Unexpected error during training: {str(e)}")
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
            print(f"Model saved in: models/{disease}/{version}/")
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