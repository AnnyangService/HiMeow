import os
import json
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import IncrementalPCA  # IncrementalPCA 사용
import joblib

# JSON 데이터 로드 함수
def load_json_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 이미지 로드 및 전처리 함수 (크기 조정)
def load_image(image_path, target_size=(64, 64)):  # 이미지 크기를 64x64로 줄임
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img).flatten()  # 이미지를 1D 배열로 평탄화
    return img_array

# 이미지 및 라벨로 데이터셋 생성 함수
def create_dataset(root_dir):
    images = []
    labels = []
    diseases = []  # 질병 이름 저장할 리스트
    genders = []  # 성별 저장할 리스트
    ages = []  # 나이 저장할 리스트
    eye_positions = []  # 눈 위치 저장할 리스트

    # 루트 디렉토리의 각 하위 폴더에서 파일 처리
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(subdir, file)
                image_path = json_path.replace('.json', '.jpg')  # JSON과 짝을 이루는 이미지 경로

                if os.path.exists(image_path):
                    # JSON 파일에서 데이터 추출
                    data = load_json_data(json_path)

                    # 'meta' 필드가 images 안에 존재
                    meta_data = data.get('images', {}).get('meta', {})  # 'images' -> 'meta'

                    # 질병 정보 추출
                    disease_name = data['label']['label_disease_nm']
                    disease_lv_1 = data['label']['label_disease_lv_1']
                    disease_lv_2 = data['label']['label_disease_lv_2']
                    disease_lv_3 = data['label']['label_disease_lv_3']  # 확정 라벨

                    # 성별, 나이, 눈 위치 정보 추출
                    gender = meta_data.get('gender', 0)
                    age = meta_data.get('age', 0)

                    # 눈 위치를 숫자로 변환
                    eye_position_str = meta_data.get('eye_position', 'Unknown')
                    if eye_position_str == '왼쪽눈':
                        eye_position = 0
                    elif eye_position_str == '오른쪽눈':
                        eye_position = 1
                    else:
                        eye_position = -1  # 알 수 없는 경우

                    # 라벨에 확정 라벨에 더 큰 가중치 부여
                    if disease_lv_3 != 'null':
                        final_label = disease_lv_3
                    elif disease_lv_2 != 'null':
                        final_label = disease_lv_2
                    else:
                        final_label = disease_lv_1

                    # 이미지 로드 및 전처리
                    image_array = load_image(image_path)

                    # 데이터셋에 이미지, 라벨, 질병 이름, 성별, 나이, 눈 위치 추가
                    images.append(image_array)
                    labels.append(final_label)
                    diseases.append(disease_name)
                    genders.append(gender)
                    ages.append(age)
                    eye_positions.append(eye_position)

    return np.array(images), labels, genders, ages, eye_positions

# IncrementalPCA를 사용하여 차원 축소하는 함수 (배치로 처리)
def reduce_image_dimensionality_batch(X_images, n_components=100, batch_size=500):
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)  # IncrementalPCA 사용
    X_images_reduced = ipca.fit_transform(X_images)
    return X_images_reduced

# 라벨 데이터 전처리
def preprocess_labels(labels):
    le = LabelEncoder()
    all_labels_encoded = le.fit_transform(labels)  # 문자열을 숫자로 변환
    return all_labels_encoded, le

def main():
    # 데이터 경로 설정 (루트 디렉토리 설정)
    root_dir = '../../../../data/Training'  # 실제 데이터 경로로 변경 (Training 폴더 안에 각 폴더가 있음)

    # 데이터셋 생성 및 전처리
    X_images, labels, genders, ages, eye_positions = create_dataset(root_dir)

    # 배치로 차원 축소
    X_images_reduced = reduce_image_dimensionality_batch(X_images, n_components=100, batch_size=500)

    # 추가적인 메타데이터와 이미지 데이터를 결합
    meta_features = np.column_stack((genders, ages, eye_positions))  # 성별, 나이, 눈 위치를 하나로 결합
    X_combined = np.hstack((X_images_reduced, meta_features))  # 차원 축소된 이미지와 메타데이터 결합

    # 라벨 전처리
    y, label_encoder = preprocess_labels(labels)

    # 학습 및 테스트 데이터로 분할
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

    # 스케일링 (이미지와 메타데이터 범위가 다르므로 스케일링 필수)a
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 랜덤 포레스트 모델 학습
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # 모델 성능 평가
    y_pred = clf.predict(X_test)
    print("정확도(Accuracy):", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # 모델 저장
    model_filename = 'random_forest_model_with_ipca.pkl'
    joblib.dump(clf, model_filename)
    print(f"{model_filename} 모델 저장 완료\n")

if __name__ == "__main__":
    main()
