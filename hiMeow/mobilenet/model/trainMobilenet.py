import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import time

class trainMobilenet(nn.Module):
    def __init__(self, num_classes=2, num_aux_features=3):
        super(trainMobilenet, self).__init__()
        self.mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

        # 모델 대부분의 레이어를 고정 (fine-tuning)
        for param in self.mobilenet.features[:-4].parameters():
            param.requires_grad = False

        in_features = self.mobilenet.classifier[1].in_features

        self.aux_features_layer = nn.Sequential(
            nn.Linear(num_aux_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features + 64, num_classes)
        )

    def forward(self, x, aux_features):
        x = self.mobilenet.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)

        aux = self.aux_features_layer(aux_features)

        combined = torch.cat((x, aux), dim=1)
        return self.classifier(combined)


def train_model(model, dataloader, optimizer, criterion, device, num_epochs=5):
    """
    모델 학습 함수
    Args:
        model: 학습할 모델
        dataloader: 데이터 로더
        optimizer: 옵티마이저
        criterion: 손실 함수
        device: 학습 디바이스
        num_epochs: 학습 에폭 수
    """
    start_time = time.time()
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, gender, age, eye_position, _, labels) in enumerate(dataloader):
            batch_start_time = time.time()

            inputs = inputs.to(device)
            aux_features = torch.stack([gender, age, eye_position], dim=1).float().to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, aux_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            batch_time = time.time() - batch_start_time
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], '
                      f'Loss: {loss.item():.4f}, Batch Time: {batch_time:.2f}s')

        avg_loss = running_loss / len(dataloader)
        epoch_time = time.time() - start_time
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}, '
              f'Epoch Time: {epoch_time:.2f}s')

        # 최고 손실 갱신
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f'New best loss: {best_loss:.4f}')

    total_time = time.time() - start_time
    print(f'Total Training Time: {total_time:.2f}s')
    return model