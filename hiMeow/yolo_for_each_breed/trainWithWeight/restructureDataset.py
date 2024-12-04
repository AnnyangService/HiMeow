from pathlib import Path
import shutil
import os


def restructure_dataset():
    """
    기존 디렉토리 구조를 새로운 구조로 재구성

    기존 구조:
    yolo_dataset/코리아_숏헤어/[disease]/datasets/[train|val|test]/[positive|negative]

    새로운 구조:
    KoreaShortHair/datasets/[train|val|test]/[disease]_[Positive|Negative]
    """

    # 기본 경로 설정
    base_path = Path('../../../yolo_dataset/코리아_숏헤어')
    new_base = Path('../../../KoreaShortHair/datasets')

    # 질병 목록
    diseases = ['Blepharitis', 'Corneal_Ulcer', 'Corneal_Secquestrum',
                'Conjunctivitis', 'Non_Ulcerative_Keratitis']

    # 새로운 디렉토리 구조 생성
    for split in ['train', 'val', 'test']:
        for disease in diseases:
            (new_base / split / f'{disease}_Positive').mkdir(parents=True, exist_ok=True)
            (new_base / split / f'{disease}_Negative').mkdir(parents=True, exist_ok=True)

    # 파일 이동
    for disease in diseases:
        disease_path = base_path / disease / 'datasets'

        for split in ['train', 'val', 'test']:
            # Positive 이미지 이동
            src_pos = disease_path / split / 'positive'
            dst_pos = new_base / split / f'{disease}_Positive'

            # Negative 이미지 이동
            src_neg = disease_path / split / 'negative'
            dst_neg = new_base / split / f'{disease}_Negative'

            if src_pos.exists():
                for img in src_pos.glob('*'):
                    shutil.copy2(img, dst_pos)
                    print(f'Copied {img.name} to {dst_pos}')

            if src_neg.exists():
                for img in src_neg.glob('*'):
                    shutil.copy2(img, dst_neg)
                    print(f'Copied {img.name} to {dst_neg}')

    print("\nDataset restructuring completed!")
    print(f"New dataset structure created at: {new_base.absolute()}")


if __name__ == "__main__":
    restructure_dataset()