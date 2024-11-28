import os
import json
import cv2
from pathlib import Path
from tqdm import tqdm


def preprocess_dataset(input_path, output_path, disease, split):
    """
    Classification을 위한 데이터셋 전처리
    input_path: 원본 데이터 경로
    output_path: 저장할 경로
    disease: 질병명
    split: 'train' 또는 'val'
    """
    # 출력 디렉토리 생성
    for class_name in ['정상', disease]:
        os.makedirs(os.path.join(output_path, split, 'images', class_name), exist_ok=True)

    def process_files(json_files, condition):
        print(f"\n{split} - {condition} 데이터 처리:")
        print(f"처리할 파일 수: {len(json_files)}")

        for json_path in tqdm(json_files, desc=f"Processing {condition}"):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                img_filename = 'crop_' + data['images']['meta']['file_name']
                img_path = os.path.join(os.path.dirname(json_path), img_filename)

                if not os.path.exists(img_path):
                    print(f"이미지 파일 없음: {img_path}")
                    continue

                img = cv2.imread(img_path)
                if img is None:
                    print(f"이미지 로드 실패: {img_path}")
                    continue

                # 이미지 리사이즈 및 저장
                img = cv2.resize(img, (640, 640))
                class_name = '정상' if condition == '무' else disease
                output_filename = data['images']['meta']['file_name']
                output_img_path = os.path.join(output_path, split, 'images', class_name, output_filename)
                cv2.imwrite(output_img_path, img)

            except Exception as e:
                print(f"Error processing {json_path}: {str(e)}")

    # 데이터 처리
    for condition in ['유', '무']:
        condition_path = os.path.join(input_path, condition)
        json_files = list(Path(condition_path).glob('*.json'))
        process_files(json_files, condition)


def main():
    # 기본 경로 설정
    base_input_path = "../../../dataset"  # 원본 데이터셋 경로
    base_output_path = "../../../yolo_dataset/preprocessed"  # 전처리된 데이터 저장 경로

    diseases = ['각막궤양', '결막염', '안검염', '각막부골편', '비궤양성각막염']

    print("Classification 데이터셋 전처리 시작...")
    os.makedirs(base_output_path, exist_ok=True)

    for disease in diseases:
        disease_output_path = os.path.join(base_output_path, disease)

        # Training 데이터 처리
        print(f"\n=== Training 데이터 처리중: {disease} ===")
        train_input_path = os.path.join(base_input_path, "Training", disease)
        preprocess_dataset(train_input_path, disease_output_path, disease, 'train')

        # Validation 데이터 처리
        print(f"\n=== Validation 데이터 처리중: {disease} ===")
        val_input_path = os.path.join(base_input_path, "Validation", disease)
        preprocess_dataset(val_input_path, disease_output_path, disease, 'val')

        # YAML 파일 생성
        yaml_content = f"""
path: {disease_output_path}
train: train/images
val: val/images
test: test/images

nc: 2
names: ['정상', '{disease}']
"""
        yaml_path = os.path.join(disease_output_path, 'data.yaml')
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)

        print(f"\n=== Completed processing {disease} ===")
        print(f"Train data: {disease_output_path}/train/images/정상, {disease_output_path}/train/images/{disease}")
        print(f"Val data: {disease_output_path}/val/images/정상, {disease_output_path}/val/images/{disease}")


if __name__ == "__main__":
    main()