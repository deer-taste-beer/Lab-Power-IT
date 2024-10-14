import pandas as pd
import os
#엑셀 스타일 A 부터 AA DO 등으로 표기하기 위해서
def get_excel_column_letter(col_idx):
    """Convert a zero-indexed column number to an Excel column letter."""
    if col_idx < 0:
        return None
    letters = ''
    while col_idx >= 0:
        col_idx, remainder = divmod(col_idx, 26)
        letters = chr(65 + remainder) + letters
        col_idx -= 1
    return letters

# 파일 경로 설정
file_path = r"C:\Users\poip8\Desktop\code\WP project before summer24\csv\2차데이터\t2차 1초단위추출 데이터 TAG(최종)_20231023-20231029 (1).csv"
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

# CSV 파일 읽기, low_memory=False 추가, 헤더 없음
df = pd.read_csv(file_path, header=None, encoding='utf-8', low_memory=False)

# 새로운 DataFrame 생성
new_df = pd.DataFrame()

# # 첫 번째 열을 A 열로 복사
# new_df['A'] = df.iloc[:, 3]  # 3번째 열을 가정 (실제 열 번호에 맞추어 수정 필요)

# 'Tag Name' 다음 열부터 끝열까지 조건 검사
keywords = ['GT2', '#2', 'GT22', '502','22GT']
nonkeywords = ['GT1', '#1', 'GT21','501','21GT']  

columns_to_copy = []

# 2행 검사 확인
print("2행 데이터 검사:")
# 열을 순회하며 조건에 따라 열을 추가
columns_to_copy = []
for col in df.columns[0:]:
    cell_value = str(df[col].iloc[1])
    if any(keyword in cell_value for keyword in keywords):
        columns_to_copy.append(col)
    elif not any(nonkeyword in cell_value for nonkeyword in nonkeywords):
        columns_to_copy.append(col)

# 조건에 맞는 열의 알파벳 이름 출력 및 조건에 맞는 열의 개수
print("\n조건을 만족하는 열:")
for i, col in enumerate(columns_to_copy):
    col_letter = get_excel_column_letter(i)  # 'Tag Name' 다음 열의 시작 인덱스에 맞추어 +4
    print(f"{col_letter} ({col})")

print(f"\n총 {len(columns_to_copy)}개의 열이 조건을 만족합니다.")

# 조건에 맞는 열들을 새 DataFrame에 추가
for i, col in enumerate(columns_to_copy, start=1):
    new_df[f'B{i}'] = df[col]

# 새로운 파일 이름 설정 및 저장
new_file_name = 'excluded_GT2_filtered_' + os.path.basename(file_path)
new_file_path = os.path.join(desktop_path, new_file_name)
new_df.to_csv(new_file_path, index=False)

print(f"\n파일이 저장되었습니다: {new_file_path}")
