import os

# 기본 경로 설정
base_path = "../../sortedDataset"

# 파일 개수를 세는 함수
def count_files_by_label(base_path):
    # 결과 저장용 딕셔너리
    results = {}

    # 품종 폴더 순회
    for breed in os.listdir(base_path):
        breed_path = os.path.join(base_path, breed)
        if not os.path.isdir(breed_path):
            continue

        # 질병 폴더 순회
        results[breed] = {}
        for disease in os.listdir(breed_path):
            disease_path = os.path.join(breed_path, disease)
            if not os.path.isdir(disease_path):
                continue

            # negative/positive 폴더 순회
            results[breed][disease] = {"negative": 0, "positive": 0}
            for label in ["negative", "positive"]:
                label_path = os.path.join(disease_path, label)
                if os.path.exists(label_path):
                    # 파일 개수 세기
                    results[breed][disease][label] = len([
                        f for f in os.listdir(label_path) if os.path.isfile(os.path.join(label_path, f))
                    ])
    
    return results

# 결과 출력
def display_results(results):
    for breed, diseases in results.items():
        print(f"품종: {breed}")
        for disease, counts in diseases.items():
            print(f"  질병: {disease}")
            print(f"    negative: {counts['negative']}개")
            print(f"    positive: {counts['positive']}개")

# 실행
results = count_files_by_label(base_path)
display_results(results)