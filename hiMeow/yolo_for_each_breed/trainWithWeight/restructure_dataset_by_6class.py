import shutil
from pathlib import Path


def restructure_dataset(source_root, target_root):
    """
    데이터셋 구조 재구성 함수
    - Negative 클래스들을 하나의 Normal 클래스로 통합
    - Positive 클래스들의 이름을 단순화
    """
    source_root = Path(source_root)
    target_root = Path(target_root)

    # 새로운 클래스 매핑 정의
    class_mapping = {
        'Blepharitis_Positive': 'Blepharitis',
        'Conjunctivitis_Positive': 'Conjunctivitis',
        'Corneal_Secquestrum_Positive': 'Corneal_Secquestrum',
        'Corneal_Ulcer_Positive': 'Corneal_Ulcer',
        'Non_Ulcerative_Keratitis_Positive': 'Non_Ulcerative_Keratitis'
    }

    # Negative 클래스 리스트
    negative_classes = [
        'Blepharitis_Negative',
        'Conjunctivitis_Negative',
        'Corneal_Secquestrum_Negative',
        'Corneal_Ulcer_Negative',
        'Non_Ulcerative_Keratitis_Negative'
    ]

    # train, val, test 각각에 대해 처리
    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split} set...")
        split_source = source_root / 'origin_datasets' / split
        split_target = target_root / 'sorted_datasets' / split

        # 타겟 디렉토리 생성
        split_target.mkdir(parents=True, exist_ok=True)

        # Normal 클래스 디렉토리 생성
        normal_target = split_target / 'Normal'
        normal_target.mkdir(exist_ok=True)

        # Negative 클래스들의 이미지를 Normal로 통합
        for neg_class in negative_classes:
            neg_dir = split_source / neg_class
            if neg_dir.exists():
                for img_path in neg_dir.glob('*.*'):
                    if img_path.suffix.lower() in ['.jpg', '.png']:
                        shutil.copy2(img_path, normal_target / f"{neg_class}_{img_path.name}")

        # Positive 클래스들 처리
        for old_class, new_class in class_mapping.items():
            old_dir = split_source / old_class
            new_dir = split_target / new_class

            if old_dir.exists():
                new_dir.mkdir(exist_ok=True)
                for img_path in old_dir.glob('*.*'):
                    if img_path.suffix.lower() in ['.jpg', '.png']:
                        shutil.copy2(img_path, new_dir / img_path.name)

        # 각 클래스별 이미지 개수 출력
        print(f"\n{split} set image counts:")
        for class_dir in split_target.iterdir():
            if class_dir.is_dir():
                count = len(list(class_dir.glob('*.*')))
                print(f"{class_dir.name}: {count}")


if __name__ == "__main__":
    current_dir = '/content/drive/MyDrive/himeow/KoreaShortHair'

    print("Starting dataset restructuring...")
    restructure_dataset(current_dir, current_dir)
    print("\nDataset restructuring completed!")