from PIL import Image, ImageEnhance
import numpy as np

# 1. 원본 이미지 불러오기
image_path = r"C:\Users\82103\Desktop\crop_C0_3e0ac8fd-60a5-11ec-8402-0a7404972c70.jpg"  # 테스트할 이미지 경로
original_image = Image.open(image_path)

# 2. Mirroring (좌우 반전)
mirrored_image = original_image.transpose(Image.FLIP_LEFT_RIGHT)

# 3. Color Shifting (채도 증가)
enhancer = ImageEnhance.Color(original_image)
color_shifted_image = enhancer.enhance(1.5)  # 채도 증가 (1.5배)

# 4. 결과 저장 또는 시각화
original_image.s
how(title="Original Image")
mirrored_image.show(title="Mirrored Image")
color_shifted_image.show(title="Color Shifted Image")

# 저장 (옵션)
mirrored_image.save(r"C:\Users\82103\Desktop\mirrored_image.jpg")
color_shifted_image.save(r"C:\Users\82103\Desktop\color_shifted_image.jpg")
