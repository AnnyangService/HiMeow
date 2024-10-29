import torch
import os
from PIL import Image
import torch.nn.functional as F
from hiMeow.mobilenet.dataloader.dataLoader import data_transforms
from hiMeow.mobilenet.utils.utils import load_model
from hiMeow.mobilenet.utils.config import ProjectConfig


def test_eye_disease(image_name, gender, age, eye_position):
    """
    눈 이미지에 대한 질병 예측을 수행합니다.
    각 질병별로 독립적인 이진 분류를 수행합니다.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    diseases = ['각막궤양', '결막염', '안검염', '백내장', '녹내장']
    config = ProjectConfig()

    image_path = os.path.join(config.test_path, image_name)

    try:
        # 이미지 로드 및 전처리
        image = Image.open(image_path).convert('RGB')
        image_tensor = data_transforms(image).unsqueeze(0).to(device)
        aux_features = torch.tensor([[gender, age, eye_position]],
                                    dtype=torch.float32).to(device)

        tests = {}
        detected_diseases = []

        # 각 질병별로 독립적으로 검사
        for disease in diseases:
            model = load_model(disease, device=device)
            if model is None:
                continue

            model.eval()
            with torch.no_grad():
                outputs = model(image_tensor, aux_features)
                # sigmoid를 사용하여 0~1 사이의 확률값으로 변환
                probability = torch.sigmoid(outputs).item()
                tests[disease] = probability

                # 임계값(0.5)을 넘는 경우 해당 질병 존재
                if probability > 0.5:
                    detected_diseases.append(disease)

        # 결과 정렬 (확률 높은 순)
        sorted_tests = sorted(tests.items(), key=lambda x: x[1], reverse=True)

        print("\n=== 예측 결과 ===")

        if not detected_diseases:
            print("분석 결과: 정상 눈으로 판단됩니다.")
            print("\n개별 질병 확률:")
        else:
            print("주의: 다음 질병이 감지되었습니다:")
            for disease in detected_diseases:
                print(f"- {disease} (확률: {tests[disease]:.2%})")
            print("\n전체 질병 분석 결과:")

        # 모든 질병의 확률 출력
        for disease, prob in sorted_tests:
            status = "주의 필요" if prob > 0.5 else "정상 범위"
            print(f"{disease}: {prob:.2%} - {status}")

        return sorted_tests

    except FileNotFoundError:
        print(f"Error: Test 폴더에서 이미지 파일을 찾을 수 없습니다: {image_name}")
        return None
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


if __name__ == "__main__":
    print("\n=== 눈 질병 예측 시스템 ===")
    print("* 이미지는 Test 폴더에 있어야 합니다 *\n")

    image_name = input("이미지 파일 이름을 입력하세요 (예: image.jpg): ")

    while True:
        try:
            gender = int(input("성별을 입력하세요 (0: 여성, 1: 남성): "))
            if gender not in [0, 1]:
                raise ValueError
            break
        except ValueError:
            print("잘못된 입력입니다. 0 또는 1을 입력하세요.")

    while True:
        try:
            age = int(input("나이를 입력하세요: "))
            if age < 0 or age > 150:
                print("유효한 나이를 입력하세요.")
                continue
            break
        except ValueError:
            print("숫자를 입력하세요.")

    while True:
        try:
            eye_position = int(input("눈 위치를 입력하세요 (0: 오른쪽, 1: 왈쪽): "))
            if eye_position not in [0, 1]:
                raise ValueError
            break
        except ValueError:
            print("잘못된 입력입니다. 0 또는 1을 입력하세요.")

    test_eye_disease(image_name, gender, age, eye_position)