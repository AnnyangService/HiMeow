# cat_data의 폴더명을 영어로 바꾸기 위한 파일이름변경 코드

import os

# dic 이용해서 한글과 영어번역 매핑
rename_mapping = {
    "각막궤양": "Corneal_Ulcer",
    "각막부골편": "Corneal_Secquestrum",
    "결막염": "Conjunctivitis",
    "비궤양성각막염": "Non_Ulcerative_Keratitis",
    "안검염": "Blepharitis"
}

# Base 디렉토리 : 본인폴더 구조에 따라 변경필요
base_path = "../../dataset"

# directory rename하는 함수
def rename_directories(base_path, rename_mapping):
    for parent_folder in ["Training", "Validation"]:
        current_path = os.path.join(base_path, parent_folder)
        for folder_name in os.listdir(current_path):
            if folder_name in rename_mapping:
                old_path = os.path.join(current_path, folder_name)
                new_path = os.path.join(current_path, rename_mapping[folder_name])
                os.rename(old_path, new_path)
                print(f'Renamed: {old_path} -> {new_path}')

rename_directories(base_path, rename_mapping)