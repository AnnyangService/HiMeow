import os
import shutil

# 이미지 파일 확장자 목록 (필요 시 추가)
IMAGE_EXTENSIONS = ['.jpg']

# 원본 폴더와 대상 폴더 경로 설정
source_folder = r"C:\Users\82103\Desktop\Blepharitis\positive"
destination_folder = r"C:\Users\82103\Desktop\Blepharitis\positive_onlyImage"

# 대상 폴더가 존재하지 않으면 생성
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 파일 복사 작업
for filename in os.listdir(source_folder):
    # 파일 확장자 확인
    if any(filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
        source_file_path = os.path.join(source_folder, filename)
        destination_file_path = os.path.join(destination_folder, filename)
        # 복사
        shutil.copy2(source_file_path, destination_file_path)
        print(f"Copied: {filename}")

print("이미지 파일 복사가 완료되었습니다.")
