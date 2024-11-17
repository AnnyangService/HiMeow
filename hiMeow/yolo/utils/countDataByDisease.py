import json
from pathlib import Path
from collections import Counter

def count_diseases(training_path):
    diseases = ['각막궤양', '결막염', '안검염', '각막부골편', '비궤양성각막염']
    disease_counts = {disease: {'유': 0, '무': 0} for disease in diseases}

    # Training 데이터 처리
    for json_file in Path(training_path).rglob("*.json"):
        try:
            with open(json_file, encoding='utf-8') as f:
                data = json.load(f)
                disease_name = data['label']['label_disease_nm']
                disease_status = data['label']['label_disease_lv_3']

                if disease_name in diseases:
                    disease_counts[disease_name][disease_status] += 1
        except Exception as e:
            print(f"Error in file {json_file}: {str(e)}")

    # 결과 출력
    print("\n=== 질병별 전체 데이터 현황 ===")
    for disease in diseases:
        total = disease_counts[disease]['유'] + disease_counts[disease]['무']
        print(f"\n{disease}:")
        print(f"- 있음(유): {disease_counts[disease]['유']}장")
        print(f"- 없음(무): {disease_counts[disease]['무']}장")
        print(f"- 총계: {total}장")
        print(f"- 비율(유:무) = {disease_counts[disease]['유']/total*100:.1f}% : {disease_counts[disease]['무']/total*100:.1f}%")

training_path = "../../../dataset/Training"
count_diseases(training_path)