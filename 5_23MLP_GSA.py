import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from SALib.analyze import sobol
from SALib.sample import saltelli


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

# CSV 파일 경로
file_path = r"C:\Users\poip8\Desktop\code\csv\excluded+GT2_filtered_추출 데이터 TAG(샘플).csv"

# CSV 파일 로드
data = pd.read_csv(file_path, skiprows=26828, nrows=209985-26828+1, header=None)

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

# 모델 하이퍼파라미터
input_size = X_train.shape[1]
hidden_size = 20  # 은닉층의 뉴런 수
output_size = 1  # 84번째 열을 예측하므로 출력 크기는 1

model = MLP(input_size, hidden_size, output_size)
model = model.to('cuda')  # GPU 사용

# 저장된 모델 경로
model_path = r"C:\Users\poip8\Desktop\code\523 MLP\EXCLUDE AFTER Train CH HIDDEN10_epoch50_model_2024-06-07_MLP.pth"

# 모델 불러오기
model.load_state_dict(torch.load(model_path))
model.eval()

# Sobol 민감도 분석
problem = {
    'num_vars': input_size,
    'names': [f'X{i}' for i in range(1, input_size+1)],
    'bounds': [[0, 1]] * input_size
}

# 샘플링
param_values = saltelli.sample(problem, 4096)

# 모델 예측 함수
def model_predict(data):
    data_tensor = torch.tensor(data, dtype=torch.float32).to('cuda')
    with torch.no_grad():
        predictions = model(data_tensor).cpu().numpy().flatten()
    return predictions

# 예측 수행
Y_sobol = np.array([model_predict(param_values[i:i+64]) for i in range(0, param_values.shape[0], 64)]).flatten()

# 민감도 분석 수행
sobol_indices = sobol.analyze(problem, Y_sobol)

# 결과 저장
results_df = pd.DataFrame({
    'Variable': [f'X{i}' for i in range(1, input_size+1)],
    'S1': sobol_indices['S1'],
    'ST': sobol_indices['ST']
})

# 현재 날짜를 기반으로 파일 이름 생성
current_date = datetime.now().strftime('%Y-%m-%d')
results_filename = f'sobol_results_{current_date}.csv'

# 결과 저장 경로
results_save_path = fr"C:\Users\poip8\Desktop\code\523 MLP\GSA\{results_filename}"

# CSV 파일로 저장
results_df.to_csv(results_save_path, index=False)
print(f'Sobol results saved to {results_save_path}')
