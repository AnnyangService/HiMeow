# 2. 필요한 라이브러리 임포트
import os
import json
import shutil

# 3. 데이터 경로 설정
source_folder = r"C:\Users\82103\Desktop\cat_data\Training"  # 기존 데이터 폴더
destination_folder = r"C:\Users\82103\Desktop\Grouping/Training"  # 새로운 데이터 저장 폴더

# 4. 파일 복사 함수
def copy_files(source_folder, destination_folder):
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith(".json"):  # JSON 파일만 처리
                json_path = os.path.join(root, file)
                with open(json_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)

                        # JSON 데이터에서 필요한 정보 추출
                        breed = data['images']['meta']['breed']  # 품종
                        disease = data['label']['label_disease_nm']  # 질병
                        disease_status = data['label']['label_disease_lv_3']  # 최종 라벨 (유/무)
                        label_file = data['label']['label_filename']  # 라벨링된 이미지 파일 이름

                        # 새 폴더 경로 생성
                        new_folder_path = os.path.join(destination_folder, breed, disease, disease_status)
                        os.makedirs(new_folder_path, exist_ok=True)

                        # 라벨링 이미지 파일 복사
                        label_source_path = os.path.join(root, label_file)  # 라벨링 이미지 경로
                        label_dest_path = os.path.join(new_folder_path, label_file)  # 라벨링 이미지 복사 경로

                        # JSON 복사 경로
                        json_dest_path = os.path.join(new_folder_path, file)

                        # 파일 존재 여부 확인 및 복사
                        if not os.path.exists(label_source_path):
                            print(f"라벨링 이미지 누락: {label_source_path}")
                        else:
                            shutil.copy(label_source_path, label_dest_path)

                        # JSON 파일 복사
                        shutil.copy(json_path, json_dest_path)

                    except KeyError as e:
                        print(f"JSON 파일 키 오류: {file} - {e}")
                    except Exception as e:
                        print(f"오류 발생: {file} - {e}")

# 5. 함수 실행
copy_files(source_folder, destination_folder)

# 6. 완료 메시지
print("파일 복사 완료")

# 7. 결과 확인 (선택 사항)
for root, dirs, files in os.walk(destination_folder):
    print(f"폴더: {root}")
    for file in files:
        print(f"  파일: {file}")
