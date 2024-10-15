import os  # 파일 및 디렉토리 관리
import json  # JSON 파일 읽기/쓰기
import numpy as np  # 배열 및 수치 계산
from PIL import Image  # 이미지 처리
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix  # 성능 평가
from sklearn.preprocessing import LabelEncoder  # 라벨 인코딩을 위한 모듈 추가
import joblib  # 모델 불러오기
import matplotlib.pyplot as plt  # 시각화 도구
import seaborn as sns  # 시각화 도구
import matplotlib.font_manager as fm  # 폰트 설정을 위한 모듈


# 한글 폰트 설정 함수
def set_korean_font():

    # 한글 폰트 설정 (Windows의 경우 'Malgun Gothic' 사용)
    plt.rcParams['font.family'] = 'Malgun Gothic'  # 혹은 'AppleGothic' (macOS의 경우)
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호가 깨지는 것을 방지


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
    diseases = []

    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(subdir, file)
                image_path = json_path.replace('.json', '.jpg')

                if not os.path.exists(image_path):
                    continue

                data = load_json_data(json_path)
                disease_name = data['label']['label_disease_nm']
                disease_lv_1 = data['label']['label_disease_lv_1']
                disease_lv_2 = data['label']['label_disease_lv_2']
                disease_lv_3 = data['label']['label_disease_lv_3']

                if disease_lv_3 != 'null':
                    final_label = disease_lv_3
                elif disease_lv_2 != 'null':
                    final_label = disease_lv_2
                else:
                    final_label = disease_lv_1

                image_array = load_image(image_path)

                images.append(image_array)
                labels.append(final_label)
                diseases.append(disease_name)

    return np.array(images), labels, diseases


# 라벨 데이터 전처리
def preprocess_labels(labels):
    le = LabelEncoder()
    all_labels_encoded = le.fit_transform(labels)  # 문자열을 숫자로 변환
    return all_labels_encoded, le


# 모델 평가 함수
def evaluate_model(clf, X_test, y_test, label_encoder):
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"정확도(Accuracy): {accuracy:.4f}")

    print("\n분류 보고서 (Classification Report):")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # 혼동 행렬 시각화
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    return accuracy, cm


def main():
    # 한글 폰트 설정
    set_korean_font()

    # Validation 데이터 경로 설정
    validation_dir = '../../../../data/Validation'  # 실제 Validation 데이터 경로로 변경

    # Validation 데이터셋 생성 및 전처리
    X_val, labels_val, diseases_val = create_dataset(validation_dir)

    # 직접 질병 이름 리스트를 지정 (고정된 5개 질병 이름)
    unique_diseases = ['각막부골편', '결막염', '비궤양성각막염', '각막궤양', '안검염']

    # 각 질병에 대해 학습된 모델 평가
    for disease in unique_diseases:
        # 해당 질병에 속하는 Validation 데이터 필터링
        disease_indices = [i for i, d in enumerate(diseases_val) if d == disease]
        X_disease_val = X_val[disease_indices]
        y_disease_val = [labels_val[i] for i in disease_indices]

        if len(X_disease_val) == 0:
            print(f"{disease}에 대한 Validation 데이터가 없습니다.")
            continue

        # 라벨 전처리
        y_val_encoded, label_encoder = preprocess_labels(y_disease_val)

        # 학습된 모델 불러오기
        model_filename = f'random_forest_model_{disease}.pkl'
        if not os.path.exists(model_filename):
            print(f"{model_filename} 모델 파일이 존재하지 않습니다.")
            continue

        clf = joblib.load(model_filename)

        # Validation 데이터로 모델 평가
        print(f"\n== {disease} 모델 평가 ==")
        evaluate_model(clf, np.array(X_disease_val), np.array(y_val_encoded), label_encoder)


if __name__ == "__main__":
    main()
