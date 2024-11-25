import os
from PIL import Image, ImageEnhance
import random

# 1. 증강 함수 정의
def augment_image(image):
    # 기본 증강 기법
    augmented_images = []

    # Mirroring (좌우 반전)
    mirrored_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    augmented_images.append(mirrored_image)

    # Color Shifting (랜덤 채도 조정)
    enhancer = ImageEnhance.Color(image)
    color_shifted_image = enhancer.enhance(random.uniform(0.8, 1.5))  # 0.8~1.5배 랜덤
    augmented_images.append(color_shifted_image)

    # 추가 증강 기법 (밝기 조정)
    enhancer = ImageEnhance.Brightness(image)
    brightness_adjusted_image = enhancer.enhance(random.uniform(0.8, 1.5))  # 밝기 조정
    augmented_images.append(brightness_adjusted_image)

    return augmented_images

# 2. 폴더 내 이미지 개수 맞추기 함수
def balance_images_in_folder(input_folder, output_folder, target_count):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 현재 폴더 내 이미지 파일 가져오기
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    current_count = len(image_files)

    print(f"현재 이미지 개수: {current_count}개")
    print(f"목표 이미지 개수: {target_count}개")

    # 이미지 개수가 이미 목표 개수 이상인 경우
    if current_count >= target_count:
        print("이미지 개수가 충분합니다. 증강을 건너뜁니다.")
        return

    # 부족한 개수 계산
    needed_count = target_count - current_count
    print(f"추가로 생성해야 할 이미지 개수: {needed_count}개")

    # 증강 작업 시작
    new_images_count = 0
    while new_images_count < needed_count:
        for file_name in image_files:
            if new_images_count >= needed_count:
                break

            # 이미지 불러오기
            image_path = os.path.join(input_folder, file_name)
            image = Image.open(image_path)

            # 증강 이미지 생성 및 저장
            augmented_images = augment_image(image)
            for aug_image in augmented_images:
                if new_images_count >= needed_count:
                    break
                output_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_aug_{new_images_count}.jpg")
                aug_image.save(output_path)
                new_images_count += 1

    print(f"증강 완료! 총 {target_count}개의 이미지가 '{output_folder}'에 저장되었습니다.")

# 3. 실행 파라미터 설정
input_folder = r"C:\Users\82103\Desktop\Blepharitis\negative" # 원본 이미지 폴더 경로
output_folder = r"C:\Users\82103\Desktop\Blepharitis\negative_agument"  # 증강된 이미지 저장 경로
target_count = 498  # 목표 이미지 개수

# 4. 실행
balance_images_in_folder(input_folder, output_folder, target_count)
