# 기존 AI-hub에서 받은 고양이 질병의 데이터 셋 구조를 아래 양식으로 변경하는 코드
# 

import os
import json
import shutil

# 기본 경로 설정
base_path = "C:/Users/82103/Desktop/cat_data"
output_path = "C:/Users/82103/Desktop/cat_data_sorted"

# 폴더 이름을 정리하는 함수
def clean_folder_name(name):
    return name.strip().replace(" ", "_")  # 공백 제거 및 밑줄로 대체

# 이미지를 품종별로 정리하는 함수
def organize_images_by_breed(base_path, output_path):
    for dataset in ["Training", "Validation"]:
        dataset_path = os.path.join(base_path, dataset)
        if not os.path.exists(dataset_path):
            print(f"경로를 찾을 수 없습니다: {dataset_path}")
            continue
        
        # 질병 폴더 탐색
        for disease in os.listdir(dataset_path):
            disease_path = os.path.join(dataset_path, disease)
            if not os.path.isdir(disease_path):
                continue

            # positive/negative 폴더 탐색
            for label in ["positive", "negative"]:
                label_path = os.path.join(disease_path, label)
                if not os.path.exists(label_path):
                    continue

                # 이미지와 JSON 파일 처리
                for file_name in os.listdir(label_path):
                    if file_name.endswith(".json"):
                        json_path = os.path.join(label_path, file_name)

                        # JSON 파일에서 품종 정보 추출
                        with open(json_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            breed = data["images"]["meta"]["breed"]
                            breed = clean_folder_name(breed)  # 폴더 이름 정리

                        # 품종/질병/레이블에 해당하는 디렉토리 생성
                        breed_dir = os.path.join(output_path, breed, disease, label)
                        os.makedirs(breed_dir, exist_ok=True)

                        # 이미지 파일 복사
                        image_file = file_name.replace(".json", ".jpg")
                        image_path = os.path.join(label_path, image_file)
                        if os.path.exists(image_path):
                            shutil.copy(image_path, os.path.join(breed_dir, image_file))
                            print(f"복사됨: {image_path} -> {breed_dir}")
                        else:
                            print(f"이미지 파일을 찾을 수 없습니다: {image_path}")

# 함수 실행
organize_images_by_breed(base_path, output_path)