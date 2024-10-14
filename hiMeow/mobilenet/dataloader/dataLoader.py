import torch
from torch.utils.data import Dataset
from torchvision import transforms
import json
import os
from PIL import Image


class CatEyeDatasetCustomized(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []

        self.eye_position_mapping = {'오른쪽눈': 0, '왼쪽눈': 1}

        print(f"Searching for JSON files in: {os.path.abspath(data_dir)}")

        for root, dirs, files in os.walk(data_dir):
            for filename in files:
                if filename.endswith('.json'):
                    file_path = os.path.join(root, filename)
                    # print(f"Processing file: {file_path}")
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)

                            # 이미지 파일명 가져오기
                            image_filename = data['label']['label_filename']
                            image_path = os.path.join(os.path.dirname(file_path), f"./{image_filename}")

                            if not os.path.exists(image_path):
                                print(f"Warning: Image file not found: {image_path}")
                                continue

                            gender = data['images']['meta']['gender']
                            age = data['images']['meta']['age']
                            eye_position = self.eye_position_mapping.get(data['images']['meta']['eye_position'], -1)

                            label_disease_nm = data['label']['label_disease_nm']
                            label_disease_lv_3 = 1 if data['label']['label_disease_lv_3'] == '유' else 0

                            self.data.append(
                                (image_path, gender, age, eye_position, label_disease_nm, label_disease_lv_3))
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON from file: {file_path}")
                    except KeyError as e:
                        print(f"KeyError in file {file_path}: {str(e)}")
                    except Exception as e:
                        print(f"Unexpected error processing file {file_path}: {str(e)}")

        print(f"Total valid samples loaded: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, gender, age, eye_position, disease_nm, disease_lv_3 = self.data[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return (
            image,
            torch.tensor(gender, dtype=torch.float),
            torch.tensor(age, dtype=torch.float),
            torch.tensor(eye_position, dtype=torch.float),
            disease_nm,
            torch.tensor(disease_lv_3, dtype=torch.long)
        )


# 데이터 변환 정의
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# # 데이터셋과 데이터 로더 생성
# dataset = CatEyeDatasetCustomized('../../../dataset/Training', transform=data_transforms)
# if len(dataset) > 0:
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
#     print(f"DataLoader created with {len(dataset)} samples.")

    # # 첫 번째 배치 확인 (선택사항)
    # images, gender, age, eye_position, disease_nm, disease_lv_3 = next(iter(dataloader))
    # print("First batch loaded successfully.")
    # print(f"Image shape: {images.shape}")
    # print(f"Gender: {gender.item()}")
    # print(f"Age: {age.item()}")
    # print(f"Eye position: {eye_position.item()}")
    # print(f"Disease name: {disease_nm[0]}")
    # print(f"Disease level 3: {disease_lv_3.item()}")
# else:
#     print("Dataset is empty. Cannot create DataLoader.")