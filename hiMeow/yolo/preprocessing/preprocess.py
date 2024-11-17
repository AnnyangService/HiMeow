import os
import json
import cv2
from pathlib import Path
from tqdm import tqdm


def preprocess_dataset(input_path, output_path, disease):
    """
    YOLO 형식에 맞게 데이터 전처리
    """

    # 출력 디렉토리 생성
    os.makedirs(os.path.join(output_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels'), exist_ok=True)

    def process_files(json_files, condition):
        print(f"\n{condition} 데이터 처리:")
        print(f"처리할 파일 수: {len(json_files)}")

        for json_path in tqdm(json_files, desc=f"Processing {condition}"):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # JSON에서 이미지 파일명을 읽고 'crop_'을 추가
                img_filename = 'crop_' + data['images']['meta']['file_name']
                img_path = os.path.join(os.path.dirname(json_path), img_filename)

                if not os.path.exists(img_path):
                    print(f"이미지 파일 없음: {img_path}")
                    continue

                img = cv2.imread(img_path)
                if img is None:
                    print(f"이미지 로드 실패: {img_path}")
                    continue

                # 출력할 때는 'crop_' 제거하고 저장
                output_filename = data['images']['meta']['file_name']
                img = cv2.resize(img, (640, 640))
                output_img_path = os.path.join(output_path, 'images', output_filename)
                cv2.imwrite(output_img_path, img)

                label_class = 0 if condition == '무' else 1
                label_content = f"{label_class} 0.5 0.5 1.0 1.0"

                label_filename = os.path.splitext(output_filename)[0] + '.txt'
                label_path = os.path.join(output_path, 'labels', label_filename)
                with open(label_path, 'w') as f:
                    f.write(label_content)

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

    print("YOLO 데이터셋 전처리 시작...")
    os.makedirs(base_output_path, exist_ok=True)

    # Training 데이터 처리
    for disease in diseases:
        print(f"\n=== Training 데이터 처리중: {disease} ===")
        train_input_path = os.path.join(base_input_path, "Training", disease)
        train_output_path = os.path.join(base_output_path, disease, "train")

        preprocess_dataset(train_input_path, train_output_path, disease)

        # Validation 데이터 처리
        print(f"\n=== Validation 데이터 처리중: {disease} ===")
        val_input_path = os.path.join(base_input_path, "Validation", disease)
        val_output_path = os.path.join(base_output_path, disease, "val")

        preprocess_dataset(val_input_path, val_output_path, disease)

        # YAML 파일 생성
        yaml_content = f"""
path: {os.path.join(base_output_path, disease)}
train: train/images
val: val/images

nc: 2
names: ['정상', '{disease}']
"""

        yaml_path = os.path.join(base_output_path, disease, 'data.yaml')
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)


if __name__ == "__main__":
    main()