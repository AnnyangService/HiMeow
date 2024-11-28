import os
import sys
from pathlib import Path
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
from tqdm import tqdm


def check_gpu():
    """GPU 사용 가능 여부 확인"""
    has_gpu = torch.cuda.is_available()
    if has_gpu:
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f}GB")
    else:
        print("No GPU available, using CPU")
    return has_gpu


def load_model(disease_name, version):
    """학습된 모델 로드"""
    model_path = Path(
        '../training/models') / disease_name / version / 'weights' / 'best.pt'

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    print(f"Loading model from {model_path}")
    model = YOLO(str(model_path))
    return model


def test_single_image(model, image_path):
    """단일 이미지 테스트"""
    try:
        results = model.predict(image_path, verbose=False)
        result = results[0]

        # 예측 결과 가져오기
        probs = result.probs.data.cpu().numpy()
        predicted_class = int(result.probs.top1)
        confidence = float(probs[predicted_class])

        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probs': probs
        }
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None


def test_model(disease_name, version):
    """모델 테스트 실행"""
    try:
        # GPU 설정
        device = 'cuda:0' if check_gpu() else 'cpu'
        print(f"Using device: {device}")

        # 모델 로드
        model = load_model(disease_name, version)
        model.to(device)

        # 테스트 데이터셋 경로 설정
        test_base = Path('../../../yolo_dataset/balanced') / disease_name / 'datasets' / 'test'
        if not test_base.exists():
            raise FileNotFoundError(f"Test directory not found: {test_base}")

        # 결과 저장을 위한 딕셔너리
        results = {
            'total': 0,
            'correct': 0,
            'predictions': []
        }

        # 하위 디렉토리 (각막궤양, 정상) 결과 저장
        class_results = {}

        print(f"\nTesting images in {test_base}")

        # test 디렉토리 내의 각 클래스 폴더 처리
        for class_dir in test_base.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                class_results[class_name] = {'total': 0, 'correct': 0}

                # 이미지 파일 처리
                for img_path in tqdm(list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')),
                                     desc=f"Testing {class_name}"):
                    prediction = test_single_image(model, str(img_path))

                    if prediction:
                        results['total'] += 1
                        class_results[class_name]['total'] += 1

                        # 예측이 맞았는지 확인
                        # 질병 클래스(각막궤양)인 경우 predicted_class가 0이어야 하고
                        # 정상 클래스인 경우 1
                        expected_class = 0 if class_name == disease_name else 1
                        is_correct = (prediction['predicted_class'] == expected_class)

                        if is_correct:
                            results['correct'] += 1
                            class_results[class_name]['correct'] += 1

                        # 결과 저장
                        results['predictions'].append({
                            'image': str(img_path),
                            'true_class': class_name,
                            'predicted_class': prediction['predicted_class'],
                            'confidence': prediction['confidence'],
                            'correct': is_correct
                        })

        # 전체 정확도 계산
        accuracy = results['correct'] / results['total'] if results['total'] > 0 else 0

        # 결과 출력
        print(f"\nTest Results for {disease_name} - {version}")
        print(f"{'=' * 50}")
        print(f"Total images tested: {results['total']}")
        print(f"Overall accuracy: {accuracy:.2%}")
        print(f"\nClass-wise Results:")

        for class_name, class_result in class_results.items():
            class_accuracy = class_result['correct'] / class_result['total'] if class_result['total'] > 0 else 0
            print(f"{class_name}:")
            print(f"  Total: {class_result['total']}")
            print(f"  Correct: {class_result['correct']}")
            print(f"  Accuracy: {class_accuracy:.2%}")

        return results

    except Exception as e:
        print(f"Test error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """메인 테스트 파이프라인"""
    if len(sys.argv) != 2:
        print("Usage: python test.py <version>")
        print("Example: python test.py v1")
        sys.exit(1)

    version = sys.argv[1]
    diseases = ['각막궤양', '결막염', '안검염', '각막부골편', '비궤양성각막염']

    print(f"Starting testing pipeline - Version {version}")

    all_results = {}
    for disease in diseases:
        print(f"\n{'=' * 50}")
        print(f"Starting testing for {disease} - {version}")
        print(f"{'=' * 50}")

        results = test_model(disease, version)
        if results:
            all_results[disease] = results
            print(f"\n=== Successfully completed testing for {disease} - {version} ===")
        else:
            print(f"\n=== Failed testing for {disease} - {version} ===")
            response = input(f"Continue with next disease? (y/n): ")
            if response.lower() != 'y':
                print("Stopping testing pipeline")
                break

    print("\nTesting pipeline completed")
    return all_results


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")