import os
import subprocess
from pathlib import Path


def train_model(disease_name, version):
    """단일 질병 모델 학습"""
    # 상대경로를 Path로 처리
    dataset_path = Path('../../../yolo_dataset/balanced') / disease_name / 'data.yaml'
    dataset_path = dataset_path.resolve()  # 절대경로로 변환

    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        return False

    model_dir = Path(f'models/{disease_name}/{version}')
    weights_dir = model_dir / 'weights'
    logs_dir = model_dir / 'logs'

    weights_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        'yolo',
        'classify',
        'train',
        'model=yolov11m-cls.pt',
        f'data={str(dataset_path)}',  # Path 객체를 문자열로 변환
        'epochs=100',
        'imgsz=640',
        'batch=16',
        'optimizer=Adam',
        'lr0=0.001',
        f'name={disease_name}_{version}',
        f'project={str(logs_dir)}'
    ]

    print(f"Starting training for {disease_name} - {version}")
    print(f"Using dataset at: {dataset_path}")  # 경로 확인용
    subprocess.run(cmd)

    best_model = logs_dir / 'best.pt'
    if best_model.exists():
        version_best = weights_dir / 'best.pt'
        os.replace(best_model, version_best)

        main_best = Path(f'models/{disease_name}/best.pt')
        if not main_best.exists():
            os.copy2(version_best, main_best)


def main():
    if len(sys.argv) != 2:
        print("Usage: python train.py <version>")
        print("Example: python train.py v1")
        sys.exit(1)

    version = sys.argv[1]
    diseases = ['각막궤양', '결막염', '안검염', '각막부골편', '비궤양성각막염']

    print(f"Starting training pipeline - Version {version}")

    for disease in diseases:
        print(f"\n=== Starting training for {disease} - {version} ===")
        train_model(disease, version)
        print(f"=== Completed training for {disease} - {version} ===\n")

        print(f"Model saved in: models/{disease}/{version}/")
        print(f"├── weights/")
        print(f"│   └── best.pt")
        print(f"├── logs/")
        print(f"└── config.yaml")


if __name__ == "__main__":
    import sys

    main()