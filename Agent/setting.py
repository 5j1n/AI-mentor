import faiss
import openai
from sentence_transformers import SentenceTransformer
import json
import chardet

def get_openai_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"  # 임베딩 생성에 적합한 최신 모델
    )
    # 첫 번째 텍스트에 대한 임베딩 벡터 반환
    embedding = response['data'][0]['embedding']
    return embedding

model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')

def get_siamese_embedding(text):
    embedding = model.encode([text])[0]
    return embedding

def detect_encoding(file_path):
    """파일의 인코딩 형식을 감지합니다."""
    with open(file_path, 'rb') as file:
        raw_data = file.read()
    return chardet.detect(raw_data)['encoding']

def load_departments_from_json(file_path):
    """JSON 파일에서 학과명을 읽어 리스트로 반환합니다."""
    departments_list = []

    # 파일 인코딩 감지
    encoding = detect_encoding(file_path)

    # JSON 파일 읽기
    try:
        with open(file_path, 'r', encoding=encoding) as json_file:
            data = json.load(json_file)

            # 각 학과명을 departments_list에 추가
            for item in data:
                # '학과명' 키를 가져와 추가
                department_name = item.get("학과명")
                if department_name:
                    departments_list.append(department_name)

    except UnicodeDecodeError as e:
        print(f"Encoding issue: {e}")
    except json.JSONDecodeError as e:
        print(f"Invalid JSON format: {e}")

    return departments_list

# 학과명 읽어오기
departments_file = 'data\department 2.json'
departments_list = load_departments_from_json(departments_file)
# FAISS 인덱스 로드
index = faiss.read_index('data\index_file_for_department 2.index')
