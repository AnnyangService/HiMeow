import random
import shutil
import cv2
import numpy as np
from pathlib import Path
from albumentations import (
    Compose, RandomRotate90, HorizontalFlip, VerticalFlip,
    RandomBrightnessContrast, HueSaturationValue,
    GaussNoise, Blur, RandomResizedCrop
)


def augment_image(image):
    """이미지 증강 함수"""
    transform = Compose([
        RandomRotate90(p=0.5),
        HorizontalFlip(p=0.5),
        RandomBrightnessContrast(p=0.5),
        HueSaturationValue(p=0.5),
        GaussNoise(p=0.3),
        Blur(blur_limit=3, p=0.3),
        RandomResizedCrop(height=image.shape[0], width=image.shape[1], scale=(0.8, 1.0), p=0.5)
    ])

    augmented = transform(image=image)
    return augmented['image']


def merge_negative_classes(source_dir):
    """Negative 클래스들을 하나의 Normal 클래스로 병합"""
    negative_classes = [
        'Blepharitis_Negative',
        'Conjunctivitis_Negative',
        'Corneal_Secquestrum_Negative',
        'Corneal_Ulcer_Negative',
        'Non_Ulcerative_Keratitis_Negative'
    ]

    images = []
    for neg_class in negative_classes:
        class_path = source_dir / neg_class
        if class_path.exists():
            images.extend(list(class_path.glob('*.jpg')) + list(class_path.glob('*.png')))

    return images


def balance_dataset(source_root, target_root, target_count=600):
    """데이터셋 균형화"""
    source_root = Path(source_root)
    target_root = Path(target_root)

    # 새로운 클래스 매핑
    new_classes = {
        'Blepharitis_Positive': 'Blepharitis',
        'Conjunctivitis_Positive': 'Conjunctivitis',
        'Corneal_Secquestrum_Positive': 'Corneal_Secquestrum',
        'Corneal_Ulcer_Positive': 'Corneal_Ulcer',
        'Non_Ulcerative_Keratitis_Positive': 'Non_Ulcerative_Keratitis',
        'Normal': 'Normal'  # 병합된 Negative 클래스들
    }

    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split} set...")
        split_dir = source_root / split

        # Normal 클래스 처리
        normal_images = merge_negative_classes(split_dir)

        for old_class, new_class in new_classes.items():
            target_class_dir = target_root / split / new_class
            target_class_dir.mkdir(parents=True, exist_ok=True)

            if old_class == 'Normal':
                source_imgs = normal_images
            else:
                class_dir = split_dir / old_class
                if not class_dir.exists():
                    continue
                source_imgs = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))

            current_count = len(source_imgs)

            if current_count > target_count:
                # 언더샘플링
                selected_imgs = random.sample(source_imgs, target_count)
                print(f"{new_class}: Under-sampling {current_count} -> {target_count}")

                for img in selected_imgs:
                    target_path = target_class_dir / img.name
                    shutil.copy2(img, target_path)
            else:
                # 오버샘플링 with augmentation
                print(f"{new_class}: Over-sampling with augmentation {current_count} -> {target_count}")

                # 원본 이미지 복사
                for img in source_imgs:
                    target_path = target_class_dir / img.name
                    shutil.copy2(img, target_path)

                # 부족한 만큼 증강된 이미지 생성
                remaining = target_count - current_count
                while remaining > 0:
                    for img in source_imgs:
                        if remaining <= 0:
                            break

                        image = cv2.imread(str(img))
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        augmented = augment_image(image)

                        target_path = target_class_dir / f"{img.stem}_aug_{remaining}{img.suffix}"
                        augmented_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(target_path), augmented_bgr)

                        remaining -= 1


if __name__ == "__main__":
    source_dataset = "/content/drive/MyDrive/himeow/KoreaShortHair/origin_datasets"
    target_dataset = "/content/drive/MyDrive/himeow/KoreaShortHair/datasets"

    print("Starting dataset balancing...")
    balance_dataset(source_dataset, target_dataset, target_count=600)
    print("\nDataset balancing completed!")

    # 결과 확인
    target_root = Path(target_dataset)
    for split in ['train', 'val', 'test']:
        print(f"\n{split} set counts:")
        for class_dir in (target_root / split).iterdir():
            if class_dir.is_dir():
                count = len(list(class_dir.glob('*')))
                print(f"{class_dir.name}: {count}")