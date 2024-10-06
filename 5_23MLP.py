import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

# 시드 설정
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 시드 값 설정
set_seed(42)


# CSV 파일 경로
file_path = r"C:\Users\poip8\Desktop\code\csv\excluded+GT2_filtered_추출 데이터 TAG(샘플).csv"

# CSV 파일 로드
data = pd.read_csv(file_path, skiprows=26828, nrows=232880-26828+1, header=None)

# 타겟 변수 (84번째 열)
Y = data.iloc[:, 83].copy()

# 입력 변수 (특정 열 제외)
data_to_scale = data.drop(columns=[83])
data_to_scale=data.drop(columns=[1,2,3,66,72,73,75,76,83,86,93,98,99,104])

# 데이터 정규화 (MinMaxScaler 사용)
scaler = MinMaxScaler()
X = scaler.fit_transform(data_to_scale)

# 데이터 분할 (훈련 및 테스트 세트)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# PyTorch Tensor로 변환
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# 데이터 로더 생성
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.leaky_relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.leaky_relu(out)
        out = self.fc2(out)
        out = self.leaky_relu(out)
        out = self.fc3(out)
        out = self.leaky_relu(out)
        out = self.fc4(out)
        out = self.leaky_relu(out)
        out = self.fc5(out)
        out = self.leaky_relu(out)
        out = self.fc6(out)
        return out

# 모델 하이퍼파라미터
input_size = X_train.shape[1]
hidden_size = 20  # 은닉층의 뉴런 수
output_size = 1  # 84번째 열을 예측하므로 출력 크기는 1

model = MLP(input_size, hidden_size, output_size)
model = model.to('cuda')  # GPU 사용

# 손실 함수 및 옵티마이저 설정
criterion = nn.MSELoss()  # 회귀 문제이므로 MSE 손실 함수 사용
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 학습
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        
        # 순전파
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 모델 평가
model.eval()
with torch.no_grad():
    total_loss = 0
    for inputs, targets in test_loader:
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader)
    print(f'Average Test Loss: {avg_loss:.4f}')

# 현재 날짜를 기반으로 파일 이름 생성
current_date = datetime.now().strftime('%Y-%m-%d')
model_filename = f'EXCLUDE AFTER Train CH HIDDEN10_epoch50_model_{current_date}_MLP.pth'

# 모델 저장 경로
save_path = fr"C:\Users\poip8\Desktop\code\523 MLP\{model_filename}"

# 모델 저장
torch.save(model.state_dict(), save_path)
print(f'Model saved to {save_path}')
