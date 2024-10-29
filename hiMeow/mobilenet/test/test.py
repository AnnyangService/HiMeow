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
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    diseases = ['각막궤양', '결막염', '안검염', '백내장', '녹내장']
    config = ProjectConfig()

    image_path = os.path.join(config.test_path, image_name)

    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = data_transforms(image).unsqueeze(0).to(device)
        aux_features = torch.tensor([[gender, age, eye_position]],
                                    dtype=torch.float32).to(device)

        tests = {}
        max_probability = 0

        for disease in diseases:
            model = load_model(disease, device=device)
            if model is None:
                continue

            model.eval()
            with torch.no_grad():
                outputs = model(image_tensor, aux_features)
                probabilities = F.softmax(outputs, dim=1)
                # 질병이 없을 확률 (0번 클래스)과 있을 확률 (1번 클래스)
                normal_prob = probabilities[0][0].item()
                disease_prob = probabilities[0][1].item()
                tests[disease] = disease_prob
                max_probability = max(max_probability, disease_prob)

        # 결과 정렬
        sorted_tests = sorted(tests.items(), key=lambda x: x[1], reverse=True)

        print("\n=== 예측 결과 ===")

        # 모든 질병의 확률이 낮은 경우 정상으로 판단
        if max_probability < 0.5:  # 임계값을 0.5로 설정
            print("분석 결과: 정상 눈으로 판단됩니다.")
            print("\n개별 질병 확률:")
        else:
            print("주의: 다음 질병의 가능성이 있습니다.")

        # 개별 확률 출력
        for disease, prob in sorted_tests:
            if prob > 0.5:
                print(f"{disease}: {prob:.2%} - 주의 필요")
            else:
                print(f"{disease}: {prob:.2%} - 정상 범위")

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
            eye_position = int(input("눈 위치를 입력하세요 (0: 오른쪽, 1: 왼쪽): "))
            if eye_position not in [0, 1]:
                raise ValueError
            break
        except ValueError:
            print("잘못된 입력입니다. 0 또는 1을 입력하세요.")

    test_eye_disease(image_name, gender, age, eye_position)