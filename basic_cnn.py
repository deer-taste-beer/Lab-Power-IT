import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self, csv_file, skiprows=26828, nrows=232880-26828+1):
        # CSV 파일을 읽어 들입니다.
        data = pd.read_csv(csv_file, skiprows=skiprows, nrows=nrows, header=None, dtype=np.float32)

        # B열 (83번 인덱스)을 분리합니다.
        B_column = data.iloc[:, 83].copy()

        # B열을 제외한 나머지 데이터를 스케일링 합니다.
        data_to_scale = data.drop(columns=[83])
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data_to_scale)

        # 슬라이딩 윈도우 방식으로 데이터를 생성합니다.
        self.image_height = 60
        self.image_width = scaled_data.shape[1]  # 스케일된 데이터의 열 수
        self.data = []
        self.targets = []

        # 슬라이딩 윈도우로 이미지 데이터와 타겟 데이터를 구성합니다.
        for start_row in range(0, len(scaled_data), 10):
            end_row = start_row + self.image_height
            if end_row > len(scaled_data):
                # 60개보다 적은 행이 남은 경우 무시
                break
            # 이미지 데이터 구성
            image = scaled_data[start_row:end_row]
            image = image.reshape(1, self.image_height, self.image_width)  # 채널 차원 추가
            self.data.append(image)
            # 타겟 데이터는 B열의 마지막 time step의 값입니다.
            target = B_column.iloc[end_row - 1]
            self.targets.append(target)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]  # 이 이미지는 이미 [1, height, width] 형태로 저장되어 있어야 함
        target = self.targets[idx]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

class BasicCNN(nn.Module):
    def __init__(self, input_height, input_width):
        super(BasicCNN, self).__init__()
        #커널 사이즈 2,2 3,3 2,2 로 변경
        #출력 채널 더 키워보자
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.bn = nn.BatchNorm2d(16)
        self.avg_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.dropout = nn.Dropout(p=0.1)  # 드롭아웃 비율 0.1 추가

        height_after_convs = self.calculate_output_dim(self.calculate_output_dim(self.calculate_output_dim(input_height, 1, 1, 1), 1, 1, 1), 2, 2, 1)
        width_after_convs = self.calculate_output_dim(self.calculate_output_dim(self.calculate_output_dim(input_width, 2, 1, 1), 2, 1, 1), 2, 2, 1)
        height_after_pool = self.calculate_output_dim(height_after_convs, 2, 2, 0)
        width_after_pool = self.calculate_output_dim(width_after_convs, 2, 2, 0)

        flat_features = height_after_pool * width_after_pool * 16
        #flat feature 수정하여 쓰
        self.fc1 = nn.Linear(9856, 642)
        self.fc2 = nn.Linear(642, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.bn(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))  # 드롭아웃 적용 후 결과를 다음 계층으로 전달
        x = self.fc2(x)
        return x

    def calculate_output_dim(self, input_dim, kernel_size, stride, padding):
        return (input_dim + 2 * padding - kernel_size) // stride + 1

class EarlyStopping:
    def __init__(self, patience=15, verbose=False, delta=0, save_path='models'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path
        self.saved_once = False

    def __call__(self, val_loss, model, epoch):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                if not self.early_stop:
                    self.early_stop = True
                    save_path = os.path.join(self.save_path, 'early_stop_model.pth')
                    if self.verbose:
                        print(f"Early stopping triggered at epoch {epoch}. Saving model with val loss: {val_loss:.6f}")
                    torch.save(model.state_dict(), save_path)
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch):
        save_path = os.path.join(self.save_path, 'best_val_loss_model.pth')
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), save_path)
        self.val_loss_min = val_loss
        self.saved_once = True

# 데이터셋 분할 함수
def create_data_loaders(dataset, batch_size):
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# 모델 평가 함수 정의
def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            predictions.extend(outputs.view(-1).tolist())
            actuals.extend(targets.tolist())

    predictions = np.array(predictions)
    actuals = np.array(actuals)
    r2 = r2_score(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)

    return r2, rmse, mae

start_time = time.time()

# 폴더 생성
model_save_path = 'MORE window EPOCH50 Hyper PA Channel and Kernel Change model_saves'
os.makedirs(model_save_path, exist_ok=True)

# 훈련과 검증 루프
epochs = 50
batch_size = 64
learning_rate = 0.001

# 모델 초기화 및 옵티마이저 설정
model = BasicCNN(input_height=60, input_width=177).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# 데이터 로더 생성
dataset = CustomDataset(r"C:\Users\poip8\Desktop\code\csv\excluded+GT2_filtered_추출 데이터 TAG(샘플).csv")
train_loader, val_loader, test_loader = create_data_loaders(dataset, batch_size)

early_stopping = EarlyStopping(patience=20, verbose=True, save_path=model_save_path)

print("Starting Now:", start_time)
for epoch in range(epochs):
    model.train()
    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets.view(-1, 1))
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, targets.view(-1, 1)).item()

    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}, Loss: {loss.item()}, Validation Loss: {val_loss}")

    # 조기 종료 조건 검사
    early_stopping(val_loss, model, epoch+1)

# 최종 모델 저장
final_model_path = os.path.join(model_save_path, 'final_epoch_model.pth')
torch.save(model.state_dict(), final_model_path)
print("Training completed. Final model saved.")

# 조기 종료 모델 로드 및 평가
early_stop_model_path = os.path.join(model_save_path, 'early_stop_model.pth')
if early_stopping.early_stop:
    model.load_state_dict(torch.load(early_stop_model_path))
    r2_early, rmse_early, mae_early = evaluate_model(model, test_loader)
    print(f"Early Stop Model - R^2: {r2_early}, RMSE: {rmse_early}, MAE: {mae_early}")

# Best validation loss 모델 로드 및 평가
best_val_model_path = os.path.join(model_save_path, 'best_val_loss_model.pth')
if early_stopping.saved_once:
    model.load_state_dict(torch.load(best_val_model_path))
    r2_best, rmse_best, mae_best = evaluate_model(model, test_loader)
    print(f"Best Validation Loss Model - R^2: {r2_best}, RMSE: {rmse_best}, MAE: {mae_best}")

# 전체 에포크 완료 모델 로드 및 평가
model.load_state_dict(torch.load(final_model_path))
r2_final, rmse_final, mae_final = evaluate_model(model, test_loader)
print(f"Final Epoch Model - R^2: {r2_final}, RMSE: {rmse_final}, MAE: {mae_final}")

# 종료 시간 기록 및 경과 시간 계산
end_time = time.time()
elapsed_time = end_time - start_time

# 경과 시간을 시간, 분, 초로 계산
hours = int(elapsed_time // 3600)  # 총 초를 3600으로 나누어 시간을 구합니다.
minutes = int((elapsed_time % 3600) // 60)  # 남은 초를 다시 60으로 나누어 분을 구합니다.
seconds = int(elapsed_time % 60)  # 남은 초를 계산합니다.

# 총 소요 시간 출력
print(f"Total training time: {hours} hours, {minutes} minutes and {seconds} seconds.")
