import os
import shutil
from pathlib import Path
import random
from collections import defaultdict


def balance_dataset(input_path, output_path, min_val_ratio=0.15, min_test_ratio=0.15):
    """
    전처리된 데이터를 train/val/test로 균형있게 분할
    """
    # 출력 디렉토리 생성
    for split in ['train', 'val', 'test']:
        for class_name in ['정상', Path(output_path).name]:
            os.makedirs(os.path.join(output_path, f'images/{split}/{class_name}'), exist_ok=True)

    # Train 데이터 수집
    train_images = defaultdict(list)
    train_path = Path(os.path.join(input_path, 'train', 'images'))
    for class_name in ['정상', Path(output_path).name]:
        class_path = train_path / class_name
        if class_path.exists():
            train_images[class_name].extend(list(class_path.glob('*.jpg')))

    # Validation 데이터 수집
    val_images = defaultdict(list)
    val_path = Path(os.path.join(input_path, 'val', 'images'))
    for class_name in ['정상', Path(output_path).name]:
        class_path = val_path / class_name
        if class_path.exists():
            val_images[class_name].extend(list(class_path.glob('*.jpg')))

    # 데이터 분포 출력
    print("\n현재 데이터 분포:")
    print("Training 데이터:")
    for k, v in train_images.items():
        print(f"{k}: {len(v)}개")
    print("\nValidation 데이터:")
    for k, v in val_images.items():
        print(f"{k}: {len(v)}개")

    def copy_files(file_list, split, class_name):
        for img_path in file_list:
            shutil.copy2(img_path, os.path.join(output_path, f'images/{split}/{class_name}', img_path.name))

    # Training 데이터 균형화
    min_train_count = min(len(train_images[k]) for k in train_images)
    balanced_train = {
        k: random.sample(v, min_train_count) for k, v in train_images.items()
    }

    # Test 셋 분할
    test_size = int(min_train_count * min_test_ratio)
    train_size = min_train_count - test_size

    final_sets = {
        'train': defaultdict(list),
        'test': defaultdict(list),
        'val': defaultdict(list)
    }

    # Training과 Test 분할
    for class_name in train_images.keys():
        random.shuffle(balanced_train[class_name])
        final_sets['train'][class_name] = balanced_train[class_name][:train_size]
        final_sets['test'][class_name] = balanced_train[class_name][train_size:]

    # Validation 데이터 균형화
    min_val_count = min(len(val_images[k]) for k in val_images)
    final_sets['val'] = {
        k: random.sample(v, min_val_count) for k, v in val_images.items()
    }

    # 파일 복사
    for split in ['train', 'test', 'val']:
        for class_name in train_images.keys():
            copy_files(final_sets[split][class_name], split, class_name)

    # 최종 통계 출력
    print("\n=== 최종 데이터 분포 ===")
    for split in ['train', 'val', 'test']:
        total = sum(len(list(Path(os.path.join(output_path, f'images/{split}/{class_name}')).glob('*.jpg')))
                   for class_name in ['정상', Path(output_path).name])
        print(f"\n{split}: 총 {total}개")
        for class_name in ['정상', Path(output_path).name]:
            count = len(list(Path(os.path.join(output_path, f'images/{split}/{class_name}')).glob('*.jpg')))
            print(f"- {class_name}: {count}개")

    # YAML 파일 생성
    yaml_content = f"""
path: {output_path}
train: images/train
val: images/val
test: images/test

nc: 2
names: ['정상', '{Path(output_path).name}']
"""

    with open(os.path.join(output_path, 'data.yaml'), 'w') as f:
        f.write(yaml_content)

    print(f"\n{Path(output_path).name} End")


def main():
    base_input_path = "../../../yolo_dataset/preprocessed"
    base_output_path = "../../../yolo_dataset/balanced"
    diseases = ['각막궤양', '결막염', '안검염', '각막부골편', '비궤양성각막염']

    print("Starting keep dataset balance...")
    os.makedirs(base_output_path, exist_ok=True)

    for disease in diseases:
        print(f"\n=== Starting balancing: {disease} ===")
        input_path = os.path.join(base_input_path, disease)
        output_path = os.path.join(base_output_path, disease)
        balance_dataset(input_path, output_path)


if __name__ == "__main__":
    main()