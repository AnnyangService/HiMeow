import os
import shutil
from pathlib import Path
import random
from collections import defaultdict


def balance_dataset(input_path, output_path, min_val_ratio=0.15, min_test_ratio=0.15):
    """
    전처리된 데이터를 train/val/test로 균형있게 분할하고 유/무 비율도 맞춤
    """
    # 출력 디렉토리 생성
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_path, f'images/{split}'), exist_ok=True)
        os.makedirs(os.path.join(output_path, f'labels/{split}'), exist_ok=True)

    # Train 데이터 수집
    train_images = defaultdict(list)
    train_img_path = Path(os.path.join(input_path, 'train', 'images'))
    for img_path in train_img_path.glob('*.jpg'):
        label_path = Path(os.path.join(input_path, 'train', 'labels', img_path.stem + '.txt'))

        with open(label_path, 'r') as f:
            label = f.read().strip()
            condition = '유' if label.startswith('1') else '무'
            train_images[condition].append((img_path, label_path))

    # Validation 데이터 수집
    val_images = defaultdict(list)
    val_img_path = Path(os.path.join(input_path, 'val', 'images'))
    for img_path in val_img_path.glob('*.jpg'):
        label_path = Path(os.path.join(input_path, 'val', 'labels', img_path.stem + '.txt'))

        with open(label_path, 'r') as f:
            label = f.read().strip()
            condition = '유' if label.startswith('1') else '무'
            val_images[condition].append((img_path, label_path))

    # 데이터 분포 출력
    print("\n현재 데이터 분포:")
    print("Training 데이터:")
    for k, v in train_images.items():
        print(f"{k}: {len(v)}개")
    print("\nValidation 데이터:")
    for k, v in val_images.items():
        print(f"{k}: {len(v)}개")

    def copy_files(file_list, split):
        for img_path, label_path in file_list:
            shutil.copy(img_path, os.path.join(output_path, f'images/{split}', img_path.name))
            shutil.copy(label_path, os.path.join(output_path, f'labels/{split}', label_path.name))

    # Training 데이터 균형화
    min_train_count = min(len(train_images['유']), len(train_images['무']))
    balanced_train = {
        '유': random.sample(train_images['유'], min_train_count),
        '무': random.sample(train_images['무'], min_train_count)
    }

    # Test 셋 분할 (균형화된 training 데이터에서)
    test_size = int(min_train_count * min_test_ratio)
    train_size = min_train_count - test_size

    final_sets = {
        'train': {'유': [], '무': []},
        'test': {'유': [], '무': []},
        'val': {'유': [], '무': []}
    }

    # Training과 Test 분할
    for condition in ['유', '무']:
        random.shuffle(balanced_train[condition])
        final_sets['train'][condition] = balanced_train[condition][:train_size]
        final_sets['test'][condition] = balanced_train[condition][train_size:]

    # Validation 데이터 균형화
    min_val_count = min(len(val_images['유']), len(val_images['무']))
    final_sets['val'] = {
        '유': random.sample(val_images['유'], min_val_count),
        '무': random.sample(val_images['무'], min_val_count)
    }

    # 파일 복사
    for split in ['train', 'test', 'val']:
        for condition in ['유', '무']:
            copy_files(final_sets[split][condition], split)

    # 최종 통계 출력
    print("\n=== 최종 데이터 분포 ===")
    for split in ['train', 'val', 'test']:
        total = len(list(Path(os.path.join(output_path, f'images/{split}')).glob('*.jpg')))
        print(f"\n{split}: 총 {total}개")

        # 유/무 개수 세기
        labels_path = Path(os.path.join(output_path, f'labels/{split}'))
        pos_count = len([1 for f in labels_path.glob('*.txt') if open(f).read().strip().startswith('1')])
        neg_count = total - pos_count
        print(f"- 유: {pos_count}개")
        print(f"- 무: {neg_count}개")

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
    # 기본 경로 설정
    base_input_path = "../../../yolo_dataset/preprocessed"  # 전처리된 데이터 경로
    base_output_path = "../../../yolo_dataset/balanced"  # 균형화된 데이터 저장 경로

    # 질병 리스트
    diseases = ['각막궤양', '결막염', '안검염', '각막부골편', '비궤양성각막염']

    print("Starting keep dataset balance...")

    # 출력 디렉토리 생성
    os.makedirs(base_output_path, exist_ok=True)

    # 각 질병별 처리
    for disease in diseases:
        print(f"\n=== Starting balancing: {disease} ===")
        input_path = os.path.join(base_input_path, disease)
        output_path = os.path.join(base_output_path, disease)
        balance_dataset(input_path, output_path)


if __name__ == "__main__":
    main()