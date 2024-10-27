import os
import sys
from pathlib import Path

class ProjectConfig:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProjectConfig, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """프로젝트 경로 초기화"""
        # utils/config.py 기준으로 프로젝트 루트 찾기
        current_file = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file)
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

        # 주요 디렉토리 설정
        self.dataset_dir = os.path.join(self.project_root, 'dataset')
        self.train_path = os.path.join(self.dataset_dir, 'Training')
        self.validation_path = os.path.join(self.dataset_dir, 'Validation')
        self.models_dir = os.path.join(self.project_root, 'models')
        self.results_dir = os.path.join(self.project_root, 'results')

        # 시스템 경로에 프로젝트 루트 추가
        if self.project_root not in sys.path:
            sys.path.append(self.project_root)

    def get_model_path(self, disease_name):
        """질병별 모델 파일 경로 반환"""
        return os.path.join(self.models_dir, f'mobilenet_v2_{disease_name}_model.pth')

    def create_directories(self):
        """필요한 디렉토리 생성"""
        directories = [
            self.dataset_dir,
            self.train_path,
            self.validation_path,
            self.models_dir,
            self.results_dir
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def __str__(self):
        """현재 설정된 경로들을 문자열로 반환"""
        return f"""
Project Configuration:
- Project Root: {self.project_root}
- Dataset Directory: {self.dataset_dir}
  ├─ Training: {self.train_path}
  └─ Validation: {self.validation_path}
- Models Directory: {self.models_dir}
- Results Directory: {self.results_dir}
"""