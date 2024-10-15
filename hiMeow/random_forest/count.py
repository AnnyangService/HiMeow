import os

# 탐색할 데이터셋 디렉토리 경로를 설정
root_dir = '../../../dataset'  # 실제 데이터셋 디렉토리 경로로 변경

def count_images_in_folders(root_dir):
    folder_info = []
    total_images = 0

    # 루트 디렉토리부터 시작해 하위 폴더와 파일을 탐색
    for subdir, dirs, files in os.walk(root_dir):
        image_count = len([file for file in files if file.endswith('.jpg') or file.endswith('.png')])
        folder_info.append((subdir, image_count))
        total_images += image_count

    return folder_info, total_images

def main():
    folder_info, total_images = count_images_in_folders(root_dir)

    # 각 폴더와 해당 폴더 안의 이미지 파일 수 출력
    print(f"총 이미지 파일 수: {total_images}\n")
    for folder, image_count in folder_info:
        print(f"폴더: {folder}, 이미지 파일 수: {image_count}")

if __name__ == "__main__":
    main()
