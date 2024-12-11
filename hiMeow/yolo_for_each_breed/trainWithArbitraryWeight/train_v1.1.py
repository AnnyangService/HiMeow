from ultralytics import YOLO
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import yaml
import shutil
import os
from google.colab import drive

# 질병 클래스 정의
DISEASES = [
    'Corneal_Ulcer_Positive', 'Corneal_Ulcer_Negative',
    'Corneal_Secquestrum_Positive', 'Corneal_Secquestrum_Negative',
    'Conjunctivitis_Positive', 'Conjunctivitis_Negative',
    'Non_Ulcerative_Keratitis_Positive', 'Non_Ulcerative_Keratitis_Negative',
    'Blepharitis_Positive', 'Blepharitis_Negative'
]

# 코리안 숏헤어의 질병별 특성과 발생 빈도를 고려한 가중치 설정
BREED_VULNERABILITY = {
    'korean_shorthair': {
        'Corneal_Ulcer': 1.2,          # 각막 궤양 - 다소 취약
        'Corneal_Secquestrum': 1.1,    # 각막 부검 - 약간 취약
        'Conjunctivitis': 1.3,         # 결막염 - 매우 취약
        'Non_Ulcerative_Keratitis': 1.4, # 비궤양성 각막염 - 가장 취약
        'Blepharitis': 1.1             # 눈꺼풀염 - 약간 취약
    }
}

# 질병별 임계값 설정
DISEASE_THRESHOLDS = {
    'korean_shorthair': {
        'Corneal_Ulcer': {
            'threshold': 0.6,
            'description': '각막 궤양은 외상이나 감염에 의해 발생하며, 코리안 숏헤어는 다소 취약한 편임'
        },
        'Corneal_Secquestrum': {
            'threshold': 0.55,
            'description': '각막 부검은 페르시안보다는 덜하지만 어느 정도 취약성 있음'
        },
        'Conjunctivitis': {
            'threshold': 0.65,
            'description': '결막염은 코리안 숏헤어에서 매우 흔하게 발생하는 질환으로 높은 주의 필요'
        },
        'Non_Ulcerative_Keratitis': {
            'threshold': 0.7,
            'description': '비궤양성 각막염은 코리안 숏헤어에서 가장 주의해야 할 안구 질환'
        },
        'Blepharitis': {
            'threshold': 0.55,
            'description': '눈꺼풀염은 어느 정도 발생 가능성이 있어 주의 필요'
        }
    }
}

def mount_drive():
    """Google Drive 마운트"""
    drive.mount('/content/drive', force_remount=True)
    print("Google Drive mounted successfully")

def setup_directories():
    """작업 디렉토리 설정"""
    temp_dataset_dir = Path('/content/temp_dataset')
    temp_dataset_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path('/content/drive/MyDrive/cat_disease_results')
    results_dir.mkdir(parents=True, exist_ok=True)

    return temp_dataset_dir, results_dir

def check_gpu():
    """GPU 사용 가능 여부 확인"""
    has_gpu = torch.cuda.is_available()
    if has_gpu:
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f}GB")
    else:
        print("No GPU available, using CPU")
    return has_gpu

def validate_dataset_structure(dataset_path):
    """데이터셋 구조 검증"""
    dataset_path = Path(dataset_path)
    required_dirs = ['train', 'val', 'test']  # test 디렉토리 추가

    # 기본 디렉토리 구조 확인
    for dir_name in required_dirs:
        dir_path = dataset_path / dir_name
        if not dir_path.exists():
            print(f"Creating {dir_name} directory...")
            dir_path.mkdir(parents=True, exist_ok=True)

    # 각 클래스별 디렉토리 확인
    for class_name in DISEASES:
        for dir_name in required_dirs:
            class_dir = dataset_path / dir_name / class_name
            if not class_dir.exists():
                print(f"Creating class directory: {class_dir}")
                class_dir.mkdir(parents=True, exist_ok=True)

def copy_dataset_from_drive():
    """Drive에서 데이터셋 복사 및 구조 검증"""
    source_path = Path('/content/drive/MyDrive/data_sorted/코리아_숏헤어')
    local_path = Path('/content/temp_dataset/datasets')

    try:
        print(f"\nChecking source path: {source_path}")
        if not source_path.exists():
            raise FileNotFoundError(f"Source dataset path not found: {source_path}")

        print("\nScanning directory structure...")

        if local_path.exists():
            print(f"Removing existing dataset directory: {local_path}")
            shutil.rmtree(local_path)

        validate_dataset_structure(local_path)

        print("\nCopying image files...")
        total_files = 0
        diseases = ['Corneal_Ulcer', 'Corneal_Secquestrum', 'Conjunctivitis',
                   'Non_Ulcerative_Keratitis', 'Blepharitis']

        for disease in diseases:
            # Positive 케이스
            pos_path = source_path / disease / "positive"
            if pos_path.exists():
                print(f"Processing {disease} Positive...")
                img_files = list(pos_path.glob('*.jpg')) + list(pos_path.glob('*.jpeg')) + \
                           list(pos_path.glob('*.png')) + list(pos_path.glob('*.JPG'))

                # 데이터 분할 비율 설정 (70:20:10)
                np.random.seed(42)
                for img_path in img_files:
                    rand_val = np.random.random()
                    if rand_val < 0.7:
                        split_dir = 'train'
                    elif rand_val < 0.9:
                        split_dir = 'val'
                    else:
                        split_dir = 'test'

                    dest_dir = local_path / split_dir / f"{disease}_Positive"
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(img_path, dest_dir / img_path.name)
                    total_files += 1

            # Negative 케이스
            neg_path = source_path / disease / "negative"
            if neg_path.exists():
                print(f"Processing {disease} Negative...")
                img_files = list(neg_path.glob('*.jpg')) + list(neg_path.glob('*.jpeg')) + \
                           list(neg_path.glob('*.png')) + list(neg_path.glob('*.JPG'))

                for img_path in img_files:
                    rand_val = np.random.random()
                    if rand_val < 0.7:
                        split_dir = 'train'
                    elif rand_val < 0.9:
                        split_dir = 'val'
                    else:
                        split_dir = 'test'

                    dest_dir = local_path / split_dir / f"{disease}_Negative"
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(img_path, dest_dir / img_path.name)
                    total_files += 1

        print(f"\nTotal files copied: {total_files}")

        if total_files == 0:
            raise ValueError("No image files found!")

        # 데이터셋 통계 출력
        print("\nDataset Statistics:")
        for split in ['train', 'val', 'test']:
            split_path = local_path / split
            if split_path.exists():
                total_images = sum(len(list(split_path.glob(f'{cls}/*.jpg'))) for cls in DISEASES)
                print(f"\n{split}: {total_images} images")

                for cls in DISEASES:
                    cls_images = len(list((split_path / cls).glob('*.jpg')))
                    print(f"  {cls}: {cls_images} images")

        return local_path

    except Exception as e:
        print(f"\nError in dataset copying: {str(e)}")
        raise

def apply_breed_weights(predictions, breed='korean_shorthair'):
    """품종별 가중치 적용"""
    weighted_predictions = predictions.copy()

    for disease, weight in BREED_VULNERABILITY[breed].items():
        pos_idx = DISEASES.index(f'{disease}_Positive')
        neg_idx = DISEASES.index(f'{disease}_Negative')

        weighted_predictions[:, pos_idx] *= weight

        total = weighted_predictions[:, pos_idx] + weighted_predictions[:, neg_idx]
        weighted_predictions[:, pos_idx] /= total
        weighted_predictions[:, neg_idx] /= total

    return weighted_predictions

def get_diagnosis(predictions, breed='korean_shorthair'):
    """질병 진단 결과 도출"""
    diagnosis = {}

    for disease in BREED_VULNERABILITY[breed].keys():
        pos_idx = DISEASES.index(f'{disease}_Positive')
        threshold = DISEASE_THRESHOLDS[breed][disease]['threshold']

        is_positive = predictions[0, pos_idx] > threshold
        confidence = predictions[0, pos_idx] if is_positive else predictions[0, pos_idx + 1]

        diagnosis[disease] = {
            'diagnosis': 'Positive' if is_positive else 'Negative',
            'confidence': float(confidence),
            'threshold_used': threshold,
            'description': DISEASE_THRESHOLDS[breed][disease]['description']
        }

    return diagnosis

def train_model(version='v1'):
    """모델 학습"""
    try:
        os.environ['WANDB_DISABLED'] = 'true'
        os.environ['WANDB_MODE'] = 'offline'

        device = 'cuda:0' if check_gpu() else 'cpu'

        _, results_dir = setup_directories()
        local_dataset = copy_dataset_from_drive()

        model = YOLO('yolov8m-cls.pt')

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        training_args = {
            'data': str(local_dataset),
            'epochs': 100,
            'batch': 32,
            'imgsz': 640,
            'device': device,
            'project': str(results_dir),
            'name': f'yolo_cls_{version}',
            'optimizer': 'SGD',
            'lr0': 0.01,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'cos_lr': True,
            'seed': 42,
            'workers': 2,
            'exist_ok': True,
            'pretrained': True,
            'amp': True,
            'patience': 50,
            'save': True,
            'save_period': 10,
            'cache': True,
            'rect': True
        }

        results = model.train(**training_args)
        return model, results

    except Exception as e:
        print(f"Training error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def validate_model(model):
    """모델 검증"""
    try:
        val_args = {
            'data': str(Path('/content/temp_dataset/datasets')),
            'device': 0 if torch.cuda.is_available() else 'cpu',
            'verbose': True,
            'split': 'val'  # validation 데이터셋 명시
        }

        results = model.val(**val_args)
        return results
    except Exception as e:
        print(f"Validation error: {str(e)}")
        return None

def test_model(model):
    """모델 테스트"""
    try:
        test_args = {
            'data': str(Path('/content/temp_dataset/datasets')),
            'device': 0 if torch.cuda.is_available() else 'cpu',
            'verbose': True,
            'split': 'test'  # test 데이터셋 명시
        }

        results = model.val(**test_args)  # YOLO의 val 메서드로 테스트 수행

        # 테스트 결과 저장
        if results is not None:
            save_dir = Path('/content/drive/MyDrive/cat_disease_results')
            results_path = save_dir / 'test_results.csv'

            metrics = {
                'top1_accuracy': results.top1,
                'top5_accuracy': results.top5,
            }

            with open(results_path, 'w') as f:
                f.write("Metric,Value\n")
                for k, v in metrics.items():
                    f.write(f"{k},{v}\n")

            print(f"Test results saved to {results_path}")

        return results
    except Exception as e:
        print(f"Test error: {str(e)}")
        return None


def main():
    try:
        mount_drive()

        version = 'v1'
        print(f"Starting training pipeline - Version {version}")

        model, train_results = train_model(version=version)

        if model is not None:
            print("Training completed!")

            print("\nRunning validation...")
            val_results = validate_model(model)
            if val_results is not None:
                print("Validation completed!")

            print("\nRunning test...")
            test_results = test_model(model)
            if test_results is not None:
                print("Testing completed!")

                # 테스트 이미지에 대한 예측 수행
                test_img = '/content/test_image.jpg'
                if os.path.exists(test_img):
                    results = model.predict(test_img)
                    predictions = results[0].probs.cpu().numpy().reshape(1, -1)

                    weighted_preds = apply_breed_weights(predictions)
                    diagnosis = get_diagnosis(weighted_preds)

                    print("\nDiagnosis Results:")
                    for disease, result in diagnosis.items():
                        print(f"\n{disease}:")
                        print(f"  Diagnosis: {result['diagnosis']}")
                        print(f"  Confidence: {result['confidence']:.2f}")
                        print(f"  Description: {result['description']}")
            else:
                print("Testing failed!")
        else:
            print("Training failed. Please check the error messages above.")

    except Exception as e:
        print(f"Error in main: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()