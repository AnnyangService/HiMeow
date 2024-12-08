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


def balance_dataset(source_root, target_root, target_count=600):
    """데이터셋 균형화"""
    source_root = Path(source_root)
    target_root = Path(target_root)

    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split} set...")

        for class_dir in (source_root / split).iterdir():
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            source_imgs = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            current_count = len(source_imgs)

            target_class_dir = target_root / split / class_name
            target_class_dir.mkdir(parents=True, exist_ok=True)

            if current_count > target_count:
                # 언더샘플링
                selected_imgs = random.sample(source_imgs, target_count)
                print(f"{class_name}: Under-sampling {current_count} -> {target_count}")

                # 선택된 이미지 복사
                for img in selected_imgs:
                    target_path = target_class_dir / img.name
                    shutil.copy2(img, target_path)

            else:
                # 오버샘플링 with augmentation
                print(f"{class_name}: Over-sampling with augmentation {current_count} -> {target_count}")

                # 먼저 원본 이미지 모두 복사
                for img in source_imgs:
                    target_path = target_class_dir / img.name
                    shutil.copy2(img, target_path)

                # 부족한 만큼 증강된 이미지 생성
                remaining = target_count - current_count
                while remaining > 0:
                    for img in source_imgs:
                        if remaining <= 0:
                            break

                        # 이미지 읽기
                        image = cv2.imread(str(img))
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                        # 이미지 증강
                        augmented = augment_image(image)

                        # 증강된 이미지 저장
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

    # 결과 확인을 위한 카운트 출력
    target_root = Path(target_dataset)
    for split in ['train', 'val', 'test']:
        print(f"\n{split} set counts:")
        for class_dir in (target_root / split).iterdir():
            if class_dir.is_dir():
                count = len(list(class_dir.glob('*')))
                print(f"{class_dir.name}: {count}")