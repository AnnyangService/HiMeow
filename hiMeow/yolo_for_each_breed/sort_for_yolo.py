# 코리아 숏헤어, 페르시안 두품종에 대해 yolo학습을 위해 train : val : test를 6:2:2로 변환하는
import os
import shutil
import random

# 기본 경로 설정
base_path = "../../sortedDataset"
output_path = "../..//yolo_dataset"  # YOLO 데이터셋 경로
split_ratio = [0.6, 0.2, 0.2]  # train:val:test 비율

# 데이터 분할 함수
def split_data_for_korean_shorthair(base_path, output_path, split_ratio):
    breed = "코리아_숏헤어"  # 코리안숏헤어만 처리
    breed_path = os.path.join(base_path, breed)
    if not os.path.isdir(breed_path):
        print(f"Breed folder not found: {breed_path}")
        return

    for disease in os.listdir(breed_path):
        disease_path = os.path.join(breed_path, disease)
        if not os.path.isdir(disease_path):
            continue

        # Positive/Negative 데이터 분할
        for label in ["positive", "negative"]:
            label_path = os.path.join(disease_path, label)
            if not os.path.exists(label_path):
                print(f"Label folder not found: {label_path}")
                continue

            # 파일 목록 가져오기
            files = [f for f in os.listdir(label_path) if os.path.isfile(os.path.join(label_path, f))]
            random.shuffle(files)  # 파일 리스트 섞기
            
            # 분할 인덱스 계산
            train_split = int(len(files) * split_ratio[0])
            val_split = train_split + int(len(files) * split_ratio[1])

            splits = {
                "train": files[:train_split],
                "val": files[train_split:val_split],
                "test": files[val_split:]
            }

            # 파일 복사
            for split, split_files in splits.items():
                split_dir = os.path.join(output_path, breed, disease, "datasets", split, label)
                os.makedirs(split_dir, exist_ok=True)

                for file in split_files:
                    src = os.path.join(label_path, file)
                    dest = os.path.join(split_dir, file)

                    try:
                        shutil.copy(src, dest)
                        print(f"Copied: {src} -> {dest}")
                    except Exception as e:
                        print(f"Error copying {src} -> {dest}: {e}")

# 실행
split_data_for_korean_shorthair(base_path, output_path, split_ratio)

