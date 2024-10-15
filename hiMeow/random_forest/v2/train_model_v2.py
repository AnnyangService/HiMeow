import os  # 파일 및 디렉토리 관리
import json  # JSON 파일 읽기/쓰기
import numpy as np  # 배열 및 수치 계산
from PIL import Image  # 이미지 처리
from sklearn.ensemble import RandomForestClassifier  # 랜덤 포레스트 모델
from sklearn.model_selection import train_test_split, GridSearchCV  # 데이터 분할 및 하이퍼파라미터 검색
from sklearn.metrics import classification_report, accuracy_score  # 성능 평가
from sklearn.preprocessing import LabelEncoder  # 라벨 인코딩을 위한 모듈 추가
import joblib  # 모델 저장


# JSON 데이터 로드 함수
def load_json_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# 이미지 로드 및 전처리 함수 (크기 조정)
def load_image(image_path, target_size=(256, 256)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img).flatten()  # 이미지를 1D 배열로 평탄화
    return img_array


# 이미지 및 라벨로 데이터셋 생성 함수
def create_dataset(root_dir):
    images = []
    labels = []
    diseases = []  # 질병 이름 저장할 리스트

    # 루트 디렉토리의 각 하위 폴더에서 파일 처리
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(subdir, file)
                image_path = json_path.replace('.json', '.jpg')  # JSON과 짝을 이루는 이미지 경로

                if os.path.exists(image_path):
                    # JSON 파일에서 데이터 추출
                    data = load_json_data(json_path)
                    disease_name = data['label']['label_disease_nm']
                    disease_lv_1 = data['label']['label_disease_lv_1']
                    disease_lv_2 = data['label']['label_disease_lv_2']
                    disease_lv_3 = data['label']['label_disease_lv_3']  # 확정 라벨

                    # 라벨에 확정 라벨에 더 큰 가중치 부여
                    if disease_lv_3 != 'null':
                        final_label = disease_lv_3
                    elif disease_lv_2 != 'null':
                        final_label = disease_lv_2
                    else:
                        final_label = disease_lv_1

                    # 이미지 로드 및 전처리
                    image_array = load_image(image_path)

                    # 데이터셋에 이미지, 라벨, 질병 이름 추가
                    images.append(image_array)
                    labels.append(final_label)
                    diseases.append(disease_name)

    return np.array(images), labels, diseases


# 라벨 데이터 전처리
def preprocess_labels(labels):
    le = LabelEncoder()
    all_labels_encoded = le.fit_transform(labels)  # 문자열을 숫자로 변환
    return all_labels_encoded, le


def main():
    # 데이터 경로 설정 (루트 디렉토리 설정)
    root_dir = '../../../../data/Training'  # 실제 데이터 경로로 변경 (Training 폴더 안에 각 폴더가 있음)

    # 데이터셋 생성 및 전처리
    X, labels, diseases = create_dataset(root_dir)

    # 직접 질병 이름 리스트를 지정 (고정된 5개 질병 이름)
    unique_diseases = ['각막부골편', '결막염', '비궤양성각막염', '각막궤양', '안검염']

    # 각 질병에 대해 모델을 개별적으로 학습
    for disease in unique_diseases:
        # 해당 질병에 속하는 데이터 필터링
        disease_indices = [i for i, d in enumerate(diseases) if d == disease]
        X_disease = X[disease_indices]
        y_disease = [labels[i] for i in disease_indices]

        # 라벨 전처리
        y, label_encoder = preprocess_labels(y_disease)

        # 학습 및 테스트 데이터로 분할
        X_train, X_test, y_train, y_test = train_test_split(X_disease, y, test_size=0.2, random_state=42)

        # 하이퍼파라미터 그리드 설정
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt'],
            'bootstrap': [True, False]
        }

        # 랜덤 포레스트 모델 생성
        clf = RandomForestClassifier(random_state=42)

        # GridSearchCV로 하이퍼파라미터 튜닝
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        # 최적의 하이퍼파라미터로 학습된 모델
        best_clf = grid_search.best_estimator_

        # 모델 성능 평가
        y_pred = best_clf.predict(X_test)
        print(f"질병: {disease}")
        print("정확도(Accuracy):", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

        # 모델 저장 (질병 이름을 포함한 파일명으로 저장)
        model_filename = f'random_forest_model_{disease}_tuned.pkl'
        joblib.dump(best_clf, model_filename)
        print(f"{model_filename} 모델 저장 완료\n")


if __name__ == "__main__":
    main()
