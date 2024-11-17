import json
from pathlib import Path
from collections import Counter


def count_breeds(training_path, validation_path):
    training_counter = Counter()
    validation_counter = Counter()

    # Training 데이터 카운트
    for json_file in Path(training_path).rglob("*.json"):
        try:
            with open(json_file, encoding='utf-8') as f:
                data = json.load(f)
                breed = data['images']['meta']['breed']
                training_counter[breed] += 1
        except Exception as e:
            print(f"Error in training file {json_file}: {str(e)}")

    # Validation 데이터 카운트
    for json_file in Path(validation_path).rglob("*.json"):
        try:
            with open(json_file, encoding='utf-8') as f:
                data = json.load(f)
                breed = data['images']['meta']['breed']
                validation_counter[breed] += 1
        except Exception as e:
            print(f"Error in validation file {json_file}: {str(e)}")

    print("\n=== Training 데이터 ===")
    for breed, count in training_counter.most_common():
        print(f"{breed}: {count}")
    print(f"Training 총계: {sum(training_counter.values())}")

    print("\n=== Validation 데이터 ===")
    for breed, count in validation_counter.most_common():
        print(f"{breed}: {count}")
    print(f"Validation 총계: {sum(validation_counter.values())}")

    # 공통 품종에 대한 비교
    common_breeds = set(training_counter.keys()) & set(validation_counter.keys())
    print("\n=== 공통 품종 비교 ===")
    for breed in sorted(common_breeds):
        print(f"{breed}: Training {training_counter[breed]} / Validation {validation_counter[breed]}")

training_path = "../../../dataset/Training"
validation_path = "../../../dataset/Validation"

count_breeds(training_path, validation_path)