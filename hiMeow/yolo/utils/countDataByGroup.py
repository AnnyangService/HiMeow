import json
from pathlib import Path
from collections import Counter


def count_breeds(training_path):
    # 그룹 정의
    groups = {
        '둥근 얼굴형 숏헤어': ['코리아 숏헤어', '브리티쉬 숏헤어', '아메리칸 숏헤어'],
        '납작한 얼굴형': ['페르시안', '엑조틱 숏헤어', '히말라얀'],
        '스코티시 계열': ['스코티시 폴드', '스코티시 스트레이트', '셀커크 렉스'],
        '긴 얼굴형 동양계': ['샴(샤미즈)', '이집션 마우'],
        '북유럽 장모종': ['터키시 앙고라', '노르웨이 숲', '메인 쿤', '시베리안', '랙돌'],
        '특수 체형': ['먼치킨', '스핑크스', '데본 렉스', '아메리칸 컬'],
        '날씬한 체형': ['러시안 블루', '아비시니안', '벵갈']
    }

    diseases = ['각막궤양', '결막염', '안검염', '각막부골편', '비궤양성각막염']

    # 데이터 카운터 초기화
    training_counts = {group: {disease: {'유': 0, '무': 0} for disease in diseases} for group in groups}

    # Training 데이터 처리
    for json_file in Path(training_path).rglob("*.json"):
        try:
            with open(json_file, encoding='utf-8') as f:
                data = json.load(f)
                breed = data['images']['meta']['breed']
                disease_name = data['label']['label_disease_nm']
                disease_status = data['label']['label_disease_lv_3']

                if disease_name in diseases:
                    for group_name, breeds in groups.items():
                        if breed in breeds:
                            training_counts[group_name][disease_name][disease_status] += 1
                            break
        except Exception as e:
            print(f"Error in training file {json_file}: {str(e)}")

    # 결과 출력
    print("\n=== 그룹별 질병 상세 현황 ===")
    for group in groups:
        print(f"\n{group} 그룹:")
        print("\nTraining:")
        for disease in diseases:
            print(f"\n{disease}:")
            print(f"- 있음(유): {training_counts[group][disease]['유']}장")
            print(f"- 없음(무): {training_counts[group][disease]['무']}장")




training_path = "../../../dataset/Training"

count_breeds(training_path)
