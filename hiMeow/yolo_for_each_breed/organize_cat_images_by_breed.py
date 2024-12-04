# 기존 AI-hub에서 받은 고양이 질병의 데이터 셋 구조를 아래 양식으로 변경하는 코드
# 

import os
import json
import shutil

# 기본 경로 설정
base_path = "../../dataset"
output_path = "../../sortedDataset"

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

        for disease in os.listdir(dataset_path):
            disease_path = os.path.join(dataset_path, disease)
            if not os.path.isdir(disease_path):
                continue

            for label in ["positive", "negative"]:
                label_path = os.path.join(disease_path, label)
                if not os.path.exists(label_path):
                    continue

                for file_name in os.listdir(label_path):
                    if file_name.endswith(".json"):
                        try:
                            json_path = os.path.join(label_path, file_name)

                            # JSON 파일 읽기 시도
                            try:
                                with open(json_path, "r", encoding="utf-8") as f:
                                    data = json.load(f)
                                    breed = data["images"]["meta"]["breed"]
                                    breed = clean_folder_name(breed)
                            except json.JSONDecodeError as e:
                                print(f"JSON 파일 오류 ({json_path}): {str(e)}")
                                continue
                            except KeyError as e:
                                print(f"키를 찾을 수 없음 ({json_path}): {str(e)}")
                                continue

                            # 디렉토리 생성 및 파일 복사
                            breed_dir = os.path.join(output_path, breed, disease, label)
                            os.makedirs(breed_dir, exist_ok=True)

                            image_file = file_name.replace(".json", ".jpg")
                            image_path = os.path.join(label_path, image_file)
                            if os.path.exists(image_path):
                                shutil.copy(image_path, os.path.join(breed_dir, image_file))
                                print(f"복사됨: {image_path} -> {breed_dir}")
                            else:
                                print(f"이미지 파일을 찾을 수 없습니다: {image_path}")

                        except Exception as e:
                            print(f"파일 처리 중 오류 발생 ({file_name}): {str(e)}")
                            continue

# 함수 실행
organize_images_by_breed(base_path, output_path)