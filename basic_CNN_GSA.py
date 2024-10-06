import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from SALib.sample import latin
from SALib.analyze import sobol
from sklearn.preprocessing import MinMaxScaler

# 모델 정의

class BasicCNN(nn.Module):
    def __init__(self, input_height, input_width):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 2), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(1, 2), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(2, 2), stride=(2, 2), padding=(1, 1))
        self.bn = nn.BatchNorm2d(128)
        self.avg_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.dropout = nn.Dropout(p=0.1)  # 드롭아웃 비율 0.1 추가

        self.height_after_convs = self.calculate_output_dim(self.calculate_output_dim(self.calculate_output_dim(input_height, 1, 1, 1), 1, 1, 1), 2, 2, 1)
        self.width_after_convs = self.calculate_output_dim(self.calculate_output_dim(self.calculate_output_dim(input_width, 2, 1, 1), 2, 1, 1), 2, 2, 1)
        self.height_after_pool = self.calculate_output_dim(self.height_after_convs, 2, 2, 0)
        self.width_after_pool = self.calculate_output_dim(self.width_after_convs, 2, 2, 0)

        flat_features = self.height_after_pool * self.width_after_pool * 128
        self.fc1 = nn.Linear(flat_features, 512)
        self.fc2 = nn.Linear(512, 1)

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

# 데이터 로드 및 샘플 생성 함수
# def create_samples_and_windows(csv_file, num_samples, image_height, num_vars):
#     data = pd.read_csv(csv_file, skiprows=26828, nrows=255774-26828+1)
#     # 84번째 열을 제외하고 사용할 열 이름을 가져옵니다.
#     column_names = [col for i, col in enumerate(data.columns) if i != 83]  # 0부터 시작하므로 83번째 열이 84번째 열입니다.
#     problem = {'num_vars': num_vars, 'names': column_names, 'bounds': [[0, 1]] * num_vars}
#     windows = []

#     for start in range(0, len(data), 100):  # 10-row segments
#         segment = data.iloc[start:start+100]
#         if len(segment) < 100:
#             continue  # Skip incomplete segments

#         # Perform latin hypercube sampling on the segment
#         samples = latin.sample(problem, num_samples)
#         new_df = pd.DataFrame(samples, columns=problem['names'])
#         scaler = MinMaxScaler()
#         scaled_data = scaler.fit_transform(new_df)

#         # Create sliding windows
#         for start_row in range(len(scaled_data) - image_height + 1):
#             end_row = start_row + image_height
#             image = scaled_data[start_row:end_row].reshape(1, image_height, num_vars)
#             windows.append(image)

#     return np.array(windows, dtype=np.float32), problem

def create_samples_and_windows(csv_file, num_samples, image_height, num_vars):
    data = pd.read_csv(csv_file, skiprows=26828, nrows=255774-26828+1)
    column_names = [col for i, col in enumerate(data.columns) if i != 83]  # 83번째 열은 제외
    problem = {'num_vars': num_vars, 'names': column_names, 'bounds': [[0, 1]] * num_vars}
    windows = []

    # 샘플링 로직 (기존 로직과 동일하게 유지)
    for start in range(0, len(data), 100):
        segment = data.iloc[start:start+100]
        if len(segment) < 100:
            continue
        samples = latin.sample(problem, num_samples)
        new_df = pd.DataFrame(samples, columns=problem['names'])
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(new_df)

        for start_row in range(len(scaled_data) - image_height + 1):
            end_row = start_row + image_height
            image = scaled_data[start_row:end_row].reshape(1, image_height, num_vars)
            windows.append(image)

    return np.array(windows, dtype=np.float32), problem  # 여러 윈도우와 문제 정의 반환


# Sobol 분석 수행
def perform_sobol_analysis(model, windows, problem, device):
    model.to(device)
    model.eval()
    outputs = []

    with torch.no_grad():
        for window in windows:
            window_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(1).to(device)
            output = model(window_tensor)
            outputs.append(output.item())

    # 모델 출력의 수가 예상 샘플 수와 일치하는지 확인
    num_vars = problem['num_vars']
    num_samples = len(outputs)  # 이 값은 예상된 샘플 수와 일치해야 함
    expected_samples = 1000 * (num_vars + 2)  # 첫 번째 차수와 총 효과 계산
    print(f"Number of outputs: {num_samples}, expected: {expected_samples}")

    if num_samples < expected_samples:
        print("Error: Insufficient number of samples for Sobol analysis.")
        return None

    # Sobol 분석 실행
    Si = sobol.analyze(problem, np.array(outputs), print_to_console=True, calc_second_order=False)
    return Si

import time

# 메인 실행 코드
if __name__ == "__main__":
    # CUDA device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()
    # Model definition and loading pre-trained weights
    model_path = r"C:\Users\poip8\Desktop\CNN\final_epoch_model.pth"
    model = BasicCNN(input_height=60, input_width=177)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Data loading and sample creation
    csv_file_path = r"C:\Users\poip8\Desktop\code\csv\excluded+GT2_filtered_추출 데이터 TAG(샘플).csv"
    num_samples = 3000  # Adjust number of samples per your memory capability
    image_height = 60
    num_vars = 177  # Number of input variables excluding the target column
    windows, problem = create_samples_and_windows(csv_file_path, num_samples, image_height, num_vars)

    # Performing Sobol sensitivity analysis
    sobol_results = perform_sobol_analysis(model, windows, problem, device)
    print("Sobol Analysis Results:")
    print(sobol_results)
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    # 경과 시간을 시간, 분, 초로 계산
    hours = int(elapsed_time // 3600)  # 총 초를 3600으로 나누어 시간을 구합니다.
    minutes = int((elapsed_time % 3600) // 60)  # 남은 초를 다시 60으로 나누어 분을 구합니다.
    seconds = int(elapsed_time % 60)  # 남은 초를 계산합니다.

    # 총 소요 시간 출력
    print(f"Total training time: {hours} hours, {minutes} minutes and {seconds} seconds.")
    
