# src/config.py
import torch  # PyTorch 라이브러리를 불러옵니다. (GPU/CPU 설정 등에 사용)
import os     # 운영체제(OS)와 관련된 기능(파일 경로 다루기 등)을 사용하기 위해 불러옵니다.

# --- 기본 경로 설정 ---
# 이 프로젝트 폴더가 어디에 있든 잘 찾아갈 수 있도록 경로를 설정합니다.
# os.path.abspath(__file__) : 현재 이 config.py 파일의 전체 경로를 알려줍니다.
# os.path.dirname(...) : 그 경로에서 폴더 이름만 가져옵니다.
# 예를 들어, config.py가 C:\project\src\config.py 에 있다면,
# os.path.dirname(os.path.abspath(__file__)) 은 C:\project\src 가 됩니다.
# 여기서 한 번 더 os.path.dirname을 하면 C:\project (프로젝트 루트 폴더)가 됩니다.
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 데이터 관련 폴더 및 파일 경로 설정
DATA_DIR = os.path.join(PROJECT_ROOT_DIR, "data") # "프로젝트폴더/data"
RAW_DATA_FILENAME = "movies.csv" # 원본 데이터 파일 이름 (우리가 직접 준비해야 할 파일)
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", RAW_DATA_FILENAME) # "프로젝트폴더/data/raw/movies.csv"
PROCESSED_DATA_FILENAME = "processed_movies_with_cls.pkl" # 전처리된 데이터 파일 이름 (나중에 프로그램이 만들 파일)
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", PROCESSED_DATA_FILENAME) # "프로젝트폴더/data/processed/..."

# 모델 관련 폴더 및 파일 경로 설정
MODEL_DIR = os.path.join(PROJECT_ROOT_DIR, "models") # "프로젝트폴더/models"
SCALER_FILENAME = "standard_scaler.joblib" # 수치 데이터 정규화 도구 저장 파일 이름
SCALER_PATH = os.path.join(MODEL_DIR, SCALER_FILENAME) # "프로젝트폴더/models/standard_scaler.joblib"
MLB_FILENAME = "mlb.joblib" # 장르 데이터 변환 도구 저장 파일 이름
MLB_PATH = os.path.join(MODEL_DIR, MLB_FILENAME) # "프로젝트폴더/models/mlb.joblib"
LANGUAGE_OHE_FILENAME = "language_onehot_encoder.joblib" # 언어용 OHE
LANGUAGE_OHE_PATH = os.path.join(MODEL_DIR, LANGUAGE_OHE_FILENAME)
MONTH_OHE_FILENAME = "month_onehot_encoder.joblib" # 월용 OHE
MONTH_OHE_PATH = os.path.join(MODEL_DIR, MONTH_OHE_FILENAME)

MODEL_WEIGHTS_FILENAME = "wide_deep_model_weights.pth" # 학습된 모델 가중치 저장 파일 이름
MODEL_WEIGHTS_PATH = os.path.join(MODEL_DIR, MODEL_WEIGHTS_FILENAME) # "프로젝트폴더/models/..."

# --- BERT 설정 ---
BERT_MODEL_NAME = 'bert-base-multilingual-cased' # 사용할 BERT 모델의 이름 (인터넷에서 다운로드)
BERT_MAX_LENGTH = 512 # BERT 모델이 한 번에 처리할 수 있는 글자(토큰) 수의 최대 한계

# --- 모델 구조 설정 ---
# WIDE_INPUT_DIM: Wide 모델 부분에 들어갈 정보의 총 개수.
#                 이 값은 데이터 전처리 후에 결정되므로, 일단 None(값 없음)으로 둡니다.
WIDE_INPUT_DIM = None
DEEP_INPUT_DIM = 768  # Deep 모델 부분에 들어갈 정보의 개수 (BERT 모델 특성에 따라 정해짐)
DEEP_HIDDEN_DIMS = [256, 128] # Deep 모델 내부의 중간 계산층(은닉층) 크기 설정
                              # [256, 128]은 첫 번째 중간층은 256개, 두 번째는 128개로 한다는 의미
DROPOUT_RATE = 0.2 # 모델 학습 시 과도한 암기를 막기 위한 장치(드롭아웃)의 비율 (20% 사용)

# --- 학습 하이퍼파라미터 (모델 학습 방법을 조절하는 값들) ---
LEARNING_RATE = 1e-4 # 학습률: 모델이 정답을 향해 얼마나 큰 걸음으로 나아갈지 정하는 값 (0.0001)
BATCH_SIZE = 16      # 배치 크기: 한 번에 학습할 데이터 묶음의 크기 (16개씩 묶어서 학습)
NUM_EPOCHS = 20      # 에포크 수: 전체 데이터를 몇 번 반복해서 학습할지 (20번 반복)
# DEVICE: 학습을 CPU에서 할지 GPU에서 할지 결정. GPU가 있으면 GPU 사용 (훨씬 빠름)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42     # 랜덤 시드: 실험 결과를 똑같이 재현하고 싶을 때 사용하는 숫자 (아무 숫자나 상관없음)

# --- 피처 이름 정의 (데이터 파일 안의 열(column) 이름들) ---
# 원본 데이터 파일에 있을 것으로 예상되는 열 이름들 (실제 파일과 일치해야 함!)
ORIG_REVENUE_COL = 'revenue' # 영화 수익 정보가 담긴 열 이름
ORIG_BUDGET_COL = 'budget'   # 영화 제작비 정보가 담긴 열 이름
ORIG_RATING_COL = 'rating'   # 영화 평점 정보가 담긴 열 이름
ORIG_RUNTIME_COL = 'runtime' # 영화 상영 시간 정보가 담긴 열 이름
ORIG_RELEASE_DATE_COL = 'release_date' # 영화 개봉일 정보가 담긴 열 이름
ORIG_GENRES_COL = 'genres'   # 영화 장르 정보가 담긴 열 이름 (예: "액션, 코미디")
ORIG_SYNOPSIS_COL = 'synopsis' # 영화 줄거리 정보가 담긴 열 이름
ORIG_TITLE_COL = 'title' # 실제 제목 컬럼명 
ORIG_KEYWORDS_COL = 'keywords' # 실제 키워드 컬럼명 
ORIG_LANGUAGE_COL = 'language' # 언어 컬럼 추가

# 원-핫 인코딩 접두사
GENRE_MLB_PREFIX = 'genre_' # MultiLabelBinarizer 사용 시 classes_로 바로 접근 가능하므로, 컬럼명 생성 시 사용
LANGUAGE_OHE_PREFIX = 'lang_'
MONTH_OHE_PREFIX = 'month_'

# 프로그램이 데이터를 처리하면서 새로 만들거나 사용할 열 이름들
TARGET_COL = 'success' # 우리가 예측하려는 목표값 ('성공' 여부)이 담길 열 이름
# 수치형 정보로 사용할 열 이름 목록 (개봉일에서 연, 월, 일을 뽑아내서 추가)
NUMERICAL_FEATURES = [ORIG_REVENUE_COL, ORIG_BUDGET_COL, ORIG_RATING_COL, ORIG_RUNTIME_COL,
                      'release_year', 'release_day']
COMBINED_TEXT_COL = 'combined_text_for_bert' # 결합된 텍스트 컬럼명
CLS_VECTOR_COL = 'cls_vector' # 영화 줄거리(시놉시스)를 숫자로 변환한 정보가 담길 열 이름

# --- 기타 ---
PREDICTION_THRESHOLD = 0.5 # 모델이 예측한 확률값이 이 값 이상이면 '성공'으로 판단