import pandas as pd
import json
import re
import random
import faiss
import numpy as np
import openai
import torch

# LangChain 관련
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

# Transformers 및 임베딩 관련
from transformers import GPT2Tokenizer, AutoTokenizer
from sentence_transformers import SentenceTransformer

openai.api_key='your_key'
# 예시 데이터프레임 필터링
file_paths = [
    '/content/drive/MyDrive/강의계획서 2024년 1학기.xlsx',
    '/content/drive/MyDrive/강의계획서 2023년 2학기.xlsx'
]

# 여러 파일을 읽어서 하나의 데이터프레임으로 결합
dataframes = []
for path in file_paths:
    try:
        df = pd.read_excel(path)
        dataframes.append(df)
    except Exception as e:
        print(f"파일을 읽는 도중 오류가 발생했습니다: {e}\n")

df_combined = pd.concat(dataframes, ignore_index=True)

# "교과목구분"이 "전공선택" 또는 "전공필수"인 행만 남기기
df_filtered = df_combined[df_combined['교과목구분'].isin(['전공선택', '전공필수'])]

def split_department_year(text):
    # 여러 개의 학과와 학년이 ','로 구분된 경우 분리
    items = text.split(',')
    departments = set()
    years = set()

    for item in items:
        # 학과명과 학년을 분리하는 정규식
        match = re.search(r"(.+?)\s*(\d+)(학년)?$", item.strip())
        if match:
            department = match.group(1).strip()
            year = match.group(2).strip()

            # 괄호와 괄호 안의 내용 제거
            department = re.sub(r'\(.*?\)', '', department).strip()

            # "학"으로 끝나면 "과"를 추가
            if department.endswith("학"):
                department += "과"
            # "학", "과", "부"로 끝나지 않으면 "학과"를 추가
            elif not department.endswith(("대","학", "과", "부")):
                department += "학과"
            # 중복된 학과명 제거를 위해 set에 추가
            departments.add(department)
            years.add(year)
        else:
            department = item.strip()

            # 괄호와 괄호 안의 내용 제거
            department = re.sub(r'\(.*?\)', '', department).strip()

            # "학"으로 끝나면 "과"를 추가
            if department.endswith("학"):
                department += "과"
            # "학", "과", "부"로 끝나지 않으면 "학과"를 추가
            elif not department.endswith(("대","학", "과", "부")):
                department += "학과"

            departments.add(department)
            years.add("N/A")  # 학년 정보가 없는 경우

    # 중복을 제거한 후 리스트로 변환하여 정렬된 문자열로 결합
    departments = ', '.join(sorted(departments))
    years = ', '.join(sorted(years))

    return pd.Series([departments, years], index=['학과', '학년'])

df_filtered[['학과', '학년']] = df_filtered['학과학년'].apply(split_department_year)
df_filtered['학과'] = df_filtered['학과'].apply(lambda x: x.replace('경제학부', '경제부'))
# 학과가 여러 개 써있는 경우, 각 학과에 대해 개별 행을 생성
expanded_rows = []

for index, row in df_filtered.iterrows():
    # 학과명에 쉼표로 구분된 여러 학과가 있는 경우
    departments = row['학과'].split(', ')
    for dept in departments:
        new_row = row.copy()  # 기존 행을 복사
        new_row['학과'] = dept  # 학과를 개별 학과로 설정
        expanded_rows.append(new_row)  # 새로운 행을 리스트에 추가

# 새로운 데이터프레임 생성
df_expanded = pd.DataFrame(expanded_rows)

# 기존 '학과학년' 칼럼과 불필요한 칼럼 제거
columns_to_remove = [
    '학과학년', '직전강의평가및CQI반영사항', '핵심역량과의관계', '소통역량', '창의역량', '인성역량',
    '실무역량', '도전역량', '문화역량', '핵심역량합계', '대표역량', '교과목간의연계성', '수업운영방향',
    '장애학생강의파일자료제공', '장애학생좌석배치', '장애학생기타', '장애학생기타상세',
    '장애학생과제제출기한연장', '장애학생대안적과제제시', '장애학생시험기간연장', '장애학생평가방법조정',
    '장애학생별도시험장소제공', '장애학생지원기타', '장애학생지원기타상세', '장애학생그외기술',
    '장애학생그외기술상세', '자율상대평가A비율', '자율상대평가AB비율', '자율상대평가C비율', '총비율',
    *[f'수업방식{i}주' for i in range(1, 17)],
    *[f'자료과제기타참고사항{i}주' for i in range(1, 17)],
    *[f'수업방식별시간온라인{i}주' for i in range(1, 17)],
    *[f'수업방식별시간오프라인{i}주' for i in range(1, 17)],
    '교재언어', '필요기자재', '교과목간의연계성',
]

df_expanded = df_expanded.drop(columns=[col for col in columns_to_remove if col in df_expanded.columns])
df_expanded = df_expanded.fillna(value="None") # 비어있는 칸은 None으로 채움

# 정수형인지 문자형인지 칼럼별로 출력 후 문자형이 아닌경우 문자형으로 변경
columns_to_check = [
    '연도', '학기', '분반', '학점', '중간', '기말', '출석', '과제물', '안전교육', '발표토론', '수업태도', '기타',
    '강의', '발표토론.1', 'PBL', '플립러닝', 'LMS활용', '실험실습', '기타.1', '학년'
]

for col in columns_to_check:
    if col in df_expanded.columns:
        dtype = df_expanded[col].dtype
        if dtype in ['float64', 'int64']:
            df_expanded[col] = df_expanded[col].astype(str)

filtered_departset = set()
for depart in df_expanded['학과']:
    # 쉼표로 분리하고 공백 제거 후 집합에 추가
    departments = [d.strip() for d in depart.split(',')]
    filtered_departset.update(departments)

filtered_lecture = set()
for lecture in df_expanded['교과목명']:
    # 쉼표로 분리하고 공백 제거 후 집합에 추가
    lectures = [d.strip() for d in lecture.split(',')]
    filtered_lecture.update(lectures)

department_lectures = {}
for index, row in df_expanded.iterrows():
    departments = [d.strip() for d in row['학과'].split(',')]
    lectures = [l.strip() for l in row['교과목명'].split(',')]
    for depart in departments:
        if depart not in department_lectures:
            department_lectures[depart] = set()
        department_lectures[depart].update(lectures)

# 유사도 계산 함수
def jaccard_similarity(set1, set2):
    intersection = set1 & set2
    union = set1 | set2
    similarity = len(intersection) / len(union) if len(union) != 0 else 0
    return similarity, intersection, set1, set2

def include(set,intersection):
    if set == intersection:
      return True
    return False


# 학과별 유사도 계산 및 그룹화
def group_similar_departments(departments, threshold=0.3):
    grouped_departments = []
    visited = set()

    for dep in departments:
        if dep not in visited:
            group = [dep]
            visited.add(dep)
            for other_dep in departments:
                if other_dep != dep and other_dep not in visited:
                    similarity, intersection, set1, set2 = jaccard_similarity(department_lectures[dep], department_lectures[other_dep])
                    if similarity >= threshold or include(set1,intersection) or include(set2,intersection):
                        print(f"병합될 학과들: {dep}와 {other_dep}")
                        print(f"유사도: {similarity}")
                        print(f"교집합(공통 강의): {intersection}")
                        print(f"{dep} 과목: {set1}")
                        print(f"{other_dep}과목: {set2}")
                        group.append(other_dep)
                        visited.add(other_dep)
            grouped_departments.append(group)
    return grouped_departments

departments = list(department_lectures.keys())
grouped_departments = group_similar_departments(departments)


# 학과 그룹화
groups = group_similar_departments(list(filtered_departset))

# 가장 긴 이름의 학과로 선택하여 합치기
def merge_departments(groups):
    merged_departments = {}
    for group in groups:
        chosen_depart = max(group, key=len)
        if(chosen_depart=="IT지능정보공학과"):
          chosen_depart = "컴퓨터인공지능학부"
        for dep in group:
            if dep != chosen_depart:
                if chosen_depart not in merged_departments:
                    merged_departments[chosen_depart] = set()
                merged_departments[chosen_depart].update(department_lectures[dep])
                if dep in department_lectures:
                    del department_lectures[dep]

    return merged_departments

# 학과 병합
merged_departments = merge_departments(groups)

# 병합된 학과 이름으로 데이터프레임 업데이트 및 병합된 학과 명과 합쳐져서 새로 받게 되는 학과명 출력
for dept, _ in merged_departments.items():
    # 병합된 학과들 이름들을 찾아 출력
    merged_dept_names = [d for group in groups for d in group if dept in group and d != dept]

    # 병합된 학과에 해당하는 행들 찾기
    rows_to_update = df_expanded['학과'].apply(lambda x: any(d in x for d in merged_dept_names))

    # 해당 문자열을 병합된 학과명(dept)으로 변경
    df_expanded.loc[rows_to_update, '학과'] = df_expanded.loc[rows_to_update, '학과'].apply(
        lambda x: dept if any(d in x for d in merged_dept_names) else x
    )

    # 병합된 학과명과 병합된 학과들 출력
    print(f"병합된 학과명: {dept} <- 병합된 학과들: {', '.join(merged_dept_names)}")

# 필터링된 데이터를 새로운 엑셀 파일로 저장
output_file_path = '/content/drive/MyDrive/filtered_and_split_file_updated.xlsx'
df_expanded.to_excel(output_file_path, index=False)
print(f"필터링되고 분리된 파일이 저장되었습니다: {output_file_path}")

# 엑셀 파일을 읽어 데이터프레임으로 변환
file_path = '/content/drive/MyDrive/filtered_and_split_file_updated.xlsx'
df = pd.read_excel(file_path)
departments_list=[]

def dataframe_to_json_for_department(df):
    json_data = []

    # '학년' 칼럼이 '3', '4'인 데이터만 필터링
    df_filtered = df[df['학년'].isin(['3', '4'])]

    # 학과별로 그룹화
    grouped = df_filtered.groupby('학과')

    # 각 학과별로 JSON 데이터 생성
    for department, group in grouped:
        # '학과' 칼럼 제외하고, 필요한 칼럼만 선택
        group = group[['교과목명']]

        # JSON 형태로 데이터 변환
        department_data = {
            "학과명": department,
            "내용": group.to_dict(orient='records')
        }

        departments_list.append(department)

        # 결과 리스트에 추가
        json_data.append(department_data)

    return json_data

structured_data = dataframe_to_json_for_department(df)
output_json_path = '/content/drive/MyDrive/structured_data_for_department1.json'

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

for idx, item in enumerate(structured_data):
    department_name = item['학과명']
    department_text = json.dumps(item['내용'], ensure_ascii=False, indent=4)
    tokenized = tokenizer(department_text, return_tensors='pt')
    token_count = len(tokenized['input_ids'][0])
    print(f"{department_name}: {token_count}")

with open(output_json_path, 'w', encoding='utf-8') as json_file:
    json.dump(structured_data, json_file, ensure_ascii=False, indent=4)
print(f"JSON 파일로 저장되었습니다: {output_json_path}\n")

# Document 객체로 변환
documents_for_department = []
for item in structured_data:
   structured_text = json.dumps(item, ensure_ascii=False, indent=4)
   document = Document(page_content=structured_text)
   documents_for_department.append(document)

# 엑셀 파일을 읽어 데이터프레임으로 변환
file_path = '/content/drive/MyDrive/filtered_and_split_file_updated.xlsx'
df = pd.read_excel(file_path)

def dataframe_to_json_for_courses(df):
    # 학년이 문자열이므로 문자열로 3, 4학년 필터링
    df = df[df['학년'].astype(str).isin(['3', '4'])]

    # 필터링된 데이터 확인
    print(f"필터링된 데이터 개수: {len(df)}")

    # 필요한 칼럼만 선택
    df = df[['교과목명', '권장선수과목', '학과', '수업목표']]

    # NaN 값을 None으로 변환
    df = df.where(pd.notnull(df), None)

    # 교과목명을 키로 하여 JSON 데이터 생성
    json_data = {
        row['교과목명']: {
            "교과목명": row['교과목명'],
            "학과": row['학과'],
            "수업목표": row['수업목표'],
        }
        for _, row in df.iterrows()
    }

    return json_data

# JSON으로 데이터 변환
structured_data = dataframe_to_json_for_courses(df)
courses_list = list(structured_data.keys())
print(len(courses_list))
# 변환된 JSON 데이터 확인
print(f"변환된 JSON 데이터 개수: {len(structured_data)}")
if structured_data:
    first_key = next(iter(structured_data))
    print(f"첫 번째 JSON 항목 예시: {first_key}: {structured_data[first_key]}")

output_json_path = '/content/drive/MyDrive/structured_data_for_courses.json'

# JSON 파일로 저장
with open(output_json_path, 'w', encoding='utf-8') as json_file:
    json.dump(structured_data, json_file, ensure_ascii=False, indent=4)
print(f"JSON 파일로 저장되었습니다: {output_json_path}\n")

# Document 객체로 변환하여 개별 강의 임베딩 생성 준비
documents_for_courses = []
for course_name, course_info in structured_data.items():
    # JSON 형식의 강의 정보를 문자열로 변환하여 Document 객체 생성
    structured_text = json.dumps({course_name: course_info}, ensure_ascii=False, indent=4)
    document = Document(page_content=structured_text)
    documents_for_courses.append(document)

# Document 리스트 출력
print(f"Document 객체 개수: {len(documents_for_courses)}")
if documents_for_courses:
    print(f"첫 번째 Document 예시: {documents_for_courses[0].page_content[:100]}")
print(courses_list)


model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')

def get_siamese_embedding(text):
    embedding = model.encode([text])[0]
    return embedding

# 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large-instruct')
model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')

# GPU 사용 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)  # 모델을 GPU로 이동

# 512개 토큰 생성 학과 수를 세기 위한 변수
total_departments = 0
departments_with_512_tokens = 0

# 각 문서에 대해 임베딩 생성 및 토큰 수 출력
embeddings_for_department = []
for idx, doc in enumerate(documents_for_department):
    # 학과명 추출
    doc_data = json.loads(doc.page_content)
    department_name = doc_data.get('학과명', 'Unknown Department')

    text = doc.page_content

    # 토큰화하여 토큰 수 확인
    tokenized = tokenizer(text, return_tensors='pt', truncation=True)

    # 토큰을 GPU로 이동
    tokenized = {key: value.to(device) for key, value in tokenized.items()}

    token_count = len(tokenized['input_ids'][0])

    # 전체 학과 수 증가
    total_departments += 1

    # 512개의 토큰을 생성한 학과 수 증가
    if token_count == 512:
        departments_with_512_tokens += 1

    # 임베딩 생성
    embedding = model.encode(text, convert_to_tensor=True).to(device)  # GPU로 임베딩 처리
    embeddings_for_department.append(embedding)

    # 학과명과 함께 토큰 수 및 임베딩 완료 메시지 출력
    print(f"Embedding for department '{department_name}' generated.")
    print(f"Department '{department_name}' token count: {token_count}")

# 512개 토큰 생성 학과의 비율 계산 및 출력
percentage = (departments_with_512_tokens / total_departments) * 100
print(f"최대 토큰 학과의 비율: {percentage:.2f}%")
print(f"총 학과 수: {total_departments}")

# 벡터 데이터베이스에 저장할 벡터를 numpy 배열로 변환
embedding_matrix = np.vstack([emb.cpu().numpy() for emb in embeddings_for_department]).astype('float32')
# 벡터 정규화 (코사인 유사도를 위해 필요)
faiss.normalize_L2(embedding_matrix)

# FAISS Index 생성 (내적 기반 Index 사용)
d = embedding_matrix.shape[1]  # 임베딩 벡터의 차원
index = faiss.IndexFlatIP(d)

# 정규화된 벡터를 Index에 추가
index.add(embedding_matrix)
print("added to db\n")

# 인덱스를 디스크에 저장
faiss.write_index(index, '/content/drive/MyDrive/index_file_for_department1.index')
print("Index saved to disk\n")
