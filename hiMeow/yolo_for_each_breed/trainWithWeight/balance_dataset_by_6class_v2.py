# v2, normal 클래스의 수를 3000개, 질병 클래스는 400, 100, 100
import shutil
import random
from pathlib import Path
import cv2
import numpy as np
from albumentations import (
    Compose, RandomRotate90, HorizontalFlip,
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


def balance_split_with_sampling(source_root, target_root):
    """데이터셋을 지정된 비율로 분할하고 언더/오버샘플링을 수행하는 함수"""
    source_root = Path(source_root)
    target_root = Path(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    # 클래스별 목표 이미지 수 정의
    class_counts = {
        'Normal': {'train': 3000, 'val': 750, 'test': 100},
        'Blepharitis': {'train': 400, 'val': 100, 'test': 100},
        'Conjunctivitis': {'train': 400, 'val': 100, 'test': 100},
        'Corneal_Secquestrum': {'train': 400, 'val': 100, 'test': 100},
        'Corneal_Ulcer': {'train': 400, 'val': 100, 'test': 100},
        'Non_Ulcerative_Keratitis': {'train': 400, 'val': 100, 'test': 100}
    }

    classes = list(class_counts.keys())

    for class_name in classes:
        print(f"\nProcessing {class_name}...")

        # 각 split에서 이미지 수집
        split_images = {}
        total_images = 0
        for split in ['train', 'val', 'test']:
            class_dir = source_root / split / class_name
            if class_dir.exists():
                images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
                split_images[split] = images
                print(f"Found {len(images)} images in {split}")
                total_images += len(images)
            else:
                split_images[split] = []
                print(f"No images found in {split}")

        print(f"Total images found for {class_name}: {total_images}")

        # 각 split 처리
        for split in ['train', 'val', 'test']:
            target_count = class_counts[class_name][split]
            target_dir = target_root / split / class_name
            target_dir.mkdir(parents=True, exist_ok=True)

            original_images = split_images[split]
            current_count = len(original_images)

            if current_count > target_count:
                # 언더샘플링
                print(f"Undersampling {class_name} {split} set: {current_count} -> {target_count}")
                selected_images = random.sample(original_images, target_count)
                for img in selected_images:
                    shutil.copy2(img, target_dir / img.name)

            else:
                # 원본 이미지 복사 후 부족한 경우 오버샘플링
                print(
                    f"Copying original images and augmenting {class_name} {split} set: {current_count} -> {target_count}")
                # 원본 이미지 모두 복사
                for img in original_images:
                    shutil.copy2(img, target_dir / img.name)

                # 부족한 만큼 augmentation
                if current_count < target_count:
                    remaining = target_count - current_count
                    aug_count = 0
                    while aug_count < remaining:
                        for img_path in original_images:
                            if aug_count >= remaining:
                                break

                            # 이미지 읽기 및 증강
                            image = cv2.imread(str(img_path))
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            augmented = augment_image(image)

                            # 증강된 이미지 저장
                            aug_name = f"{img_path.stem}_aug_{aug_count}{img_path.suffix}"
                            augmented_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(str(target_dir / aug_name), augmented_bgr)

                            aug_count += 1

            final_count = len(list(target_dir.glob('*.*')))
            print(f"{split} set final count: {final_count}")


if __name__ == "__main__":
    source_path = Path("/content/drive/MyDrive/himeow/KoreaShortHair/sorted_datasets")
    target_path = Path("/content/drive/MyDrive/himeow/KoreaShortHair/datasets")

    print("Starting dataset balancing with under/oversampling...")
    balance_split_with_sampling(source_path, target_path)

    print("\nVerifying final dataset distribution...")
    for split in ['train', 'val', 'test']:
        print(f"\n{split} set counts:")
        split_path = target_path / split
        if split_path.exists():
            for class_dir in split_path.iterdir():
                if class_dir.is_dir():
                    count = len(list(class_dir.glob('*.*')))
                    print(f"{class_dir.name}: {count}")