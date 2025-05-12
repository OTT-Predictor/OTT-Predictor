# src/preprocess.py
import pandas as pd  # 표 형태의 데이터를 다루는 데 아주 유용한 라이브러리 (엑셀 시트처럼 생각하세요)
import numpy as np   # 숫자 계산, 특히 배열(리스트와 비슷하지만 더 강력함)을 다룰 때 사용
# scikit-learn 라이브러리에서 데이터 전처리 도구들을 불러옵니다.
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, OneHotEncoder
# Hugging Face Transformers 라이브러리에서 BERT 관련 도구들을 불러옵니다.
from transformers import BertTokenizer, BertModel
import torch         # PyTorch 라이브러리
import joblib        # 파이썬 객체(여기서는 전처리 도구)를 파일로 저장하고 불러올 때 사용
import os            # 운영체제 기능 사용 (파일/폴더 경로 다루기)
import ast # ast 모듈 임포트 추가
from tqdm import tqdm # 반복 작업 시 진행 상황을 막대 형태로 보여주는 라이브러리

# 같은 폴더(src) 안에 있는 config.py 파일을 불러옵니다. (설정값 사용 목적)
from . import config

def load_raw_data(file_path):
    """원본 CSV 데이터를 로드하는 함수입니다."""
    if not os.path.exists(file_path): # 만약 지정된 경로에 파일이 없다면
        print(f"Error: Raw data file not found at {file_path}")
        # 실제 파일이 없을 경우, 설명을 위해 임시로 샘플 데이터를 만듭니다.
        # (실제 프로젝트에서는 이 부분 없이, 파일이 꼭 있어야 합니다!)
        print("Creating sample data for demonstration...")
        sample_data = { # 영화 정보를 담은 가상의 데이터 (딕셔셔너리 형태)
            config.ORIG_TITLE_COL: ['영화 A', '영화 B', '영화 C', '영화 D', '영화 E'], # title 추가
            config.ORIG_SYNOPSIS_COL: [ # config 파일에 정의된 'synopsis' 열 이름 사용
                '우주를 배경으로 한 영웅 이야기. 액션과 감동!', '작은 마을의 미스터리 스릴러. 반전 주의!',
                '두 남녀의 애틋한 로맨스 드라마.', '역사적 사건을 다룬 대서사극. 웅장함!',
                '웃음과 교훈이 있는 가족 코미디 애니메이션.'
            ],
            config.ORIG_KEYWORDS_COL: [ # keywords 추가 (문자열 리스트 형태 가정)
                "['hero', 'space', 'action']", "['mystery', 'thriller', 'small town']",
                "['romance', 'drama']", "['history', 'epic']",
                "['comedy', 'family', 'animation']"
            ],
            # 다른 열들도 config 파일에 정의된 이름으로 만듭니다.
            config.ORIG_REVENUE_COL: [200, 50, 150, 300, 100],
            config.ORIG_BUDGET_COL: [100, 60, 50, 250, 80],
            config.ORIG_RATING_COL: [8.5, 7.2, 9.1, 6.5, 8.8],
            config.ORIG_RUNTIME_COL: [120, 95, 110, 150, 90],
            config.ORIG_RELEASE_DATE_COL: ['2023-01-15', '2023-03-20', '2023-05-10', '2023-07-01', '2023-09-25'],
            config.ORIG_GENRES_COL: [
                '액션, SF', '미스터리, 스릴러', '로맨스, 드라마',
                '드라마, 전쟁', '코미디, 애니메이션, 가족'
            ],
            config.ORIG_LANGUAGE_COL: ['en', 'en', 'ko', 'en', 'ja'] # 언어 샘플 추가
            
        }
        return pd.DataFrame(sample_data) # 딕셔너리를 Pandas DataFrame(표) 형태로 변환하여 반환
    return pd.read_csv(file_path) # 파일이 있다면 CSV 파일을 읽어서 DataFrame으로 반환

def create_target_label(df):
    """'성공' 여부를 나타내는 타겟 레이블(정답 값)을 만드는 함수입니다."""
    # config 파일에 임의로 정의된 기준 (수익 > 제작비 AND 평점 > 8.0)에 따라 'success' 열을 만듭니다.
    # 조건을 만족하면 1 (성공), 아니면 0 (실패) 값을 가집니다.
    df[config.TARGET_COL] = ((df[config.ORIG_REVENUE_COL] > df[config.ORIG_BUDGET_COL]) & \
                             (df[config.ORIG_RATING_COL] > 8.0)).astype(int)
    return df # 'success' 열이 추가된 DataFrame 반환

def process_release_date_and_month_onehot(df, mode='train'): # 함수 이름 및 기능 변경
    """개봉일에서 연, 일 정보를 추출하고, 월은 원-핫 인코딩합니다."""
    month_ohe_path = config.MONTH_OHE_PATH # config에서 경로 사용
    month_cols_generated = []

    if config.ORIG_RELEASE_DATE_COL in df.columns:
        # 'release_date' 열의 문자열을 날짜 형태로 변환합니다. (오류 발생 시 날짜 아님 처리)
        df[config.ORIG_RELEASE_DATE_COL] = pd.to_datetime(df[config.ORIG_RELEASE_DATE_COL], errors='coerce')
        # 날짜에서 연, 월, 일 정보를 각각 뽑아 새 열로 만듭니다.
        df['release_year'] = df[config.ORIG_RELEASE_DATE_COL].dt.year
        df['release_month_orig'] = df[config.ORIG_RELEASE_DATE_COL].dt.month # 원-핫 인코딩 전 월
        df['release_day'] = df[config.ORIG_RELEASE_DATE_COL].dt.day
        # 만약 날짜 변환에 실패해서 빈 값(결측치)이 생기면, 임시로 -1로 채우고 정수형으로 만듭니다.
        for col in ['release_year', 'release_month_orig', 'release_day']:
            df[col] = df[col].fillna(-1).astype(int) # 결측치 -1로 (OHE 전처리 시 문제될 수 있으므로 주의)
                                                    # 또는 가장 흔한 값 등으로 채우는 것이 나을 수 있음

        # release_month_orig를 원-핫 인코딩
        # -1 값(결측)도 하나의 카테고리로 처리될 수 있음, 또는 특정 값으로 대체 후 OHE
        # 여기서는 1~12월만 유효한 카테고리로 보고, -1은 모든 OHE 컬럼에서 0이 되도록 처리 (handle_unknown='ignore')
        month_data_for_ohe = df[['release_month_orig']].copy()
        # 학습 시에는 1~12월이 모두 포함된 데이터로 fit 하는 것이 이상적
        # 만약 특정 월이 학습 데이터에 없다면, categories 인자를 명시적으로 제공해야 함
        valid_months = [[i for i in range(1, 13)]] # 1부터 12까지의 월

        if mode == 'train':
            month_ohe = OneHotEncoder(categories=valid_months, handle_unknown='ignore', sparse_output=False) # sparse=False 대신 sparse_output=False (최신 버전)
            month_encoded_arr = month_ohe.fit_transform(month_data_for_ohe)
            os.makedirs(os.path.dirname(month_ohe_path), exist_ok=True)
            joblib.dump(month_ohe, month_ohe_path)
            print(f"Month OneHotEncoder saved to {month_ohe_path}")
        elif mode == 'predict' or mode == 'evaluate':
            if not os.path.exists(month_ohe_path):
                raise FileNotFoundError(f"Month OHE not found at {month_ohe_path}.")
            month_ohe = joblib.load(month_ohe_path)
            month_encoded_arr = month_ohe.transform(month_data_for_ohe)
        else:
            raise ValueError("mode should be 'train', 'evaluate', or 'predict'")

        month_cols_generated = [f"{config.MONTH_OHE_PREFIX}{cat}" for cat in month_ohe.categories_[0]]
        month_encoded_df = pd.DataFrame(month_encoded_arr, columns=month_cols_generated, index=df.index)
        # 원-핫 인코딩된 월 데이터를 원래 DataFrame에 추가합니다.
        df = pd.concat([df, month_encoded_df], axis=1)
        # 원래의 'release_date' 열은 이제 필요 없으므로 삭제합니다.
        df = df.drop([config.ORIG_RELEASE_DATE_COL, 'release_month_orig'], axis=1, errors='ignore')

    return df, month_cols_generated

def preprocess_language_onehot(df, mode='train'):
    """언어 피처를 원-핫 인코딩하는 함수"""
    lang_ohe_path = config.LANGUAGE_OHE_PATH # config에서 경로 사용
    lang_cols_generated = []

    if config.ORIG_LANGUAGE_COL in df.columns:
        # 결측치 처리 (예: 'unknown' 또는 가장 흔한 언어로 대체)
        df[config.ORIG_LANGUAGE_COL] = df[config.ORIG_LANGUAGE_COL].fillna('unknown').astype(str)
        lang_data_for_ohe = df[[config.ORIG_LANGUAGE_COL]].copy()

        if mode == 'train':
            # categories='auto'로 하면 학습 데이터에 있는 모든 고유 언어를 사용
            # 또는 상위 N개 언어만 사용하고 나머지는 'other'로 묶을 수도 있음
            lang_ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # 처음 보는 언어는 무시 (모든 OHE 컬럼 0)
            lang_encoded_arr = lang_ohe.fit_transform(lang_data_for_ohe)
            os.makedirs(os.path.dirname(lang_ohe_path), exist_ok=True)
            joblib.dump(lang_ohe, lang_ohe_path)
            print(f"Language OneHotEncoder saved to {lang_ohe_path}")
        elif mode == 'predict' or mode == 'evaluate':
            if not os.path.exists(lang_ohe_path):
                raise FileNotFoundError(f"Language OHE not found at {lang_ohe_path}.")
            lang_ohe = joblib.load(lang_ohe_path)
            lang_encoded_arr = lang_ohe.transform(lang_data_for_ohe)
        else: raise ValueError("mode error")

        lang_cols_generated = [f"{config.LANGUAGE_OHE_PREFIX}{cat}" for cat in lang_ohe.categories_[0]]
        lang_encoded_df = pd.DataFrame(lang_encoded_arr, columns=lang_cols_generated, index=df.index)
        df = pd.concat([df, lang_encoded_df], axis=1)
        df = df.drop(config.ORIG_LANGUAGE_COL, axis=1, errors='ignore')
    return df, lang_cols_generated

def preprocess_numerical_features(df, mode='train'):
    """
    수익, 제작비, 평점 등 숫자 정보를 가진 피처(열)들을 전처리하는 함수입니다.
    숫자들의 크기(scale)를 비슷하게 맞춰주는 정규화(Standardization) 작업을 합니다.
    Args:
        df (pd.DataFrame): 처리할 데이터프레임.
        mode (str): 'train'(학습용 데이터 처리 시) 또는 'predict'/'evaluate'(예측/평가용 데이터 처리 시).
    """
    scaler_path = config.SCALER_PATH # 정규화 도구(scaler)를 저장/로드할 경로
    # 숫자 열에 빈 값(결측치)이 있다면, 일단 0으로 채웁니다. (더 좋은 방법은 평균/중앙값 등으로 채우는 것)
    df[config.NUMERICAL_FEATURES] = df[config.NUMERICAL_FEATURES].fillna(0)

    if mode == 'train': # 학습용 데이터를 처리할 때
        scaler = StandardScaler() # StandardScaler 라는 정규화 도구를 새로 만듭니다.
        # 이 도구를 사용해 NUMERICAL_FEATURES 에 있는 열들의 값들을 정규화합니다.
        # fit_transform은 데이터에 맞춰 도구를 학습시키고(fit) 바로 적용(transform)합니다.
        df[config.NUMERICAL_FEATURES] = scaler.fit_transform(df[config.NUMERICAL_FEATURES])
        # 나중에 예측할 때도 똑같은 기준으로 정규화해야 하므로, 학습된 scaler를 파일로 저장합니다.
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True) # 저장할 폴더가 없으면 만듭니다.
        joblib.dump(scaler, scaler_path) # scaler 객체를 파일로 저장
        print(f"Scaler saved to {scaler_path}")
    elif mode == 'predict' or mode == 'evaluate': # 예측/평가용 데이터를 처리할 때
        if not os.path.exists(scaler_path): # 저장된 scaler 파일이 없으면 오류
            raise FileNotFoundError(f"Scaler not found at {scaler_path}. Please run preprocessing in 'train' mode first.")
        scaler = joblib.load(scaler_path) # 저장된 scaler를 불러옵니다.
        # 불러온 scaler를 사용해 데이터를 정규화합니다. (절대 다시 fit_transform 하면 안 됩니다!)
        df[config.NUMERICAL_FEATURES] = scaler.transform(df[config.NUMERICAL_FEATURES])
    else: # mode 값이 잘못된 경우 오류 발생
        raise ValueError("mode should be 'train', 'evaluate', or 'predict'")
    return df

def preprocess_genre_features(df, mode='train'):
    """
    장르(genres) 정보를 모델이 이해할 수 있는 형태로 변환하는 함수입니다.
    "['공포', '스릴러']" 같은 문자열을 실제 리스트로 변환 후 멀티-핫 인코딩합니다.
    """
    mlb_path = config.MLB_PATH
    # 장르 열에 빈 값(결측치)이 있다면, 빈 리스트를 나타내는 문자열 '[]'로 채웁니다.
    df[config.ORIG_GENRES_COL] = df[config.ORIG_GENRES_COL].fillna('[]')

    def parse_genre_string(genre_str):
        try:
            # 문자열 형태의 리스트 "['a', 'b']" 를 실제 파이썬 리스트 ['a', 'b']로 변환
            genre_list = ast.literal_eval(genre_str)
            # 각 장르명의 앞뒤 공백 제거
            return [str(genre).strip() for genre in genre_list if str(genre).strip()]
        except (ValueError, SyntaxError):
            # 변환 실패 시 (예: 잘못된 형식의 문자열) 빈 리스트 반환
            return []

    # 'genre_list' 열을 생성합니다.
    df['genre_list'] = df[config.ORIG_GENRES_COL].apply(parse_genre_string)

    if mode == 'train':
        mlb = MultiLabelBinarizer()
        # MultiLabelBinarizer는 입력으로 리스트의 리스트를 기대합니다.
        # df['genre_list']는 이미 각 행이 장르 리스트인 Series이므로 바로 사용 가능합니다.
        genre_encoded = pd.DataFrame(mlb.fit_transform(df['genre_list']),
                                     columns=[f"{config.GENRE_MLB_PREFIX}{cls}" for cls in mlb.classes_],
                                     index=df.index)
        os.makedirs(os.path.dirname(mlb_path), exist_ok=True)
        joblib.dump(mlb, mlb_path)
        print(f"MultiLabelBinarizer saved to {mlb_path}")
        print(f"Encoded genre columns: {[f"{config.GENRE_MLB_PREFIX}{cls}" for cls in mlb.classes_]}")
    elif mode == 'predict' or mode == 'evaluate':
        if not os.path.exists(mlb_path):
            raise FileNotFoundError(f"MLB not found at {mlb_path}. Please run preprocessing in 'train' mode first.")
        mlb = joblib.load(mlb_path)
        genre_encoded = pd.DataFrame(mlb.transform(df['genre_list']),
                                     columns=[f"{config.GENRE_MLB_PREFIX}{cls}" for cls in mlb.classes_],
                                     index=df.index)
    else:
        raise ValueError("mode should be 'train', 'evaluate', or 'predict'")

    df = pd.concat([df, genre_encoded], axis=1)
    # 원본 장르 컬럼과 중간에 만든 genre_list 컬럼 삭제
    df = df.drop([config.ORIG_GENRES_COL, 'genre_list'], axis=1, errors='ignore')
    return df, [f"{config.GENRE_MLB_PREFIX}{cls}" for cls in mlb.classes_]

def combine_text_fields_for_bert(df):
    """title, synopsis, keywords를 결합하여 BERT 입력용 단일 텍스트 필드를 만듭니다."""
    sep_token_text = " [SEP] " # 실제 BERT [SEP] 토큰이 아니라, 구분을 위한 문자열

    # 각 필드의 결측치를 빈 문자열로 처리
    title = df[config.ORIG_TITLE_COL].fillna('').astype(str)
    synopsis = df[config.ORIG_SYNOPSIS_COL].fillna('').astype(str)

    # keywords는 문자열 리스트 형태이므로, 파싱 후 문자열로 join
    def parse_and_join_keywords(keywords_str):
        try:
            keywords_list = ast.literal_eval(keywords_str)
            return ", ".join([str(k).strip() for k in keywords_list if str(k).strip()])
        except (ValueError, SyntaxError):
            return "" # 파싱 실패 시 빈 문자열

    keywords_str_series = df[config.ORIG_KEYWORDS_COL].fillna('[]').apply(parse_and_join_keywords)

    # 세 필드를 결합. 필드가 비어있으면 해당 [SEP]도 불필요할 수 있으나, 일관성을 위해 유지하거나 조건부 처리
    df[config.COMBINED_TEXT_COL] = title + \
                                   sep_token_text + synopsis + \
                                   sep_token_text + keywords_str_series
    # 만약 필드가 비어있을 때 [SEP]을 중복으로 넣고 싶지 않다면, 각 필드별로 조건부 결합 필요
    # 예: combined = []
    #     if not title.empty: combined.append(title)
    #     if not synopsis.empty: combined.append(synopsis)
    #     if not keywords_str_series.empty: combined.append(keywords_str_series)
    #     df[config.COMBINED_TEXT_COL] = sep_token_text.join(combined)
    # 여기서는 간단히 모두 더하는 방식으로 처리. BERT 토크나이저가 중복 [SEP] 등을 처리할 수 있음.
    return df

def get_cls_vector_batch(texts, tokenizer, bert_model, device, max_length):
    """여러 개의 텍스트 묶음(배치)에 대해 한 번에 CLS 벡터를 추출하는 함수입니다."""
    # tokenizer: 텍스트를 BERT가 이해하는 숫자(토큰)로 변환하고, 특수 기호([CLS], [SEP])를 추가합니다.
    # padding='max_length': 모든 텍스트 길이를 max_length에 맞추기 위해 짧으면 빈 공간(패딩)을 채웁니다.
    # truncation=True: max_length보다 길면 잘라냅니다.
    # add_special_tokens=True: [CLS], [SEP] 자동 추가 (기본값)
    inputs = tokenizer(texts, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True, add_special_tokens=True)
    # 변환된 숫자들을 모델이 있는 장치(GPU 또는 CPU)로 옮깁니다.
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad(): # 기울기 계산을 하지 않도록 설정 (벡터 추출만 하므로 필요 없음, 메모리 절약)
        outputs = bert_model(**inputs) # BERT 모델에 입력을 넣어 결과를 얻습니다.
    # 결과 중 마지막 은닉 상태(last_hidden_state)에서 [CLS] 토큰에 해당하는 벡터만 가져옵니다.
    # [CLS] 토큰은 보통 문장 전체의 의미를 담고 있다고 알려져 있습니다.
    # .cpu().numpy()는 결과를 CPU로 옮기고 NumPy 배열 형태로 바꿉니다.
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

def extract_cls_vectors_from_df(df, tokenizer, bert_model, device, batch_size=16):
    """데이터프레임의 시놉시스 열 전체에 대해 CLS 벡터를 추출하는 함수입니다."""
    all_cls_vectors = [] # 추출된 CLS 벡터들을 담을 리스트
    # 시놉시스 열의 텍스트들을 리스트로 만듭니다. 빈 값(결측치)은 빈 문자열로 처리.
    combined_texts = df[config.COMBINED_TEXT_COL].fillna('').tolist()
    print("Extracting CLS vectors...")
    # tqdm: 반복 작업의 진행 상황을 막대바로 보여줍니다. (데이터가 많을 때 유용)
    # 전체 시놉시스를 batch_size 만큼씩 묶어서 처리합니다 (메모리 효율성).
    for i in tqdm(range(0, len(combined_texts), batch_size)):
        batch_texts = combined_texts[i:i+batch_size] # 현재 처리할 텍스트 묶음
        # get_cls_vector_batch 함수를 호출하여 CLS 벡터들을 얻습니다.
        batch_cls_vectors = get_cls_vector_batch(batch_texts, tokenizer, bert_model, device, config.BERT_MAX_LENGTH)
        all_cls_vectors.extend(list(batch_cls_vectors)) # 결과 리스트에 추가
    # 추출된 CLS 벡터들을 데이터프레임의 새 열(config.CLS_VECTOR_COL)로 추가합니다.
    df[config.CLS_VECTOR_COL] = all_cls_vectors
    return df

def run_preprocessing(raw_data_path, processed_data_path, mode='train'):
    """
    위에서 만든 모든 전처리 함수들을 순서대로 실행하는 메인 함수입니다.
    Args:
        raw_data_path (str): 원본 데이터 파일 경로.
        processed_data_path (str): 전처리된 데이터를 저장할 파일 경로.
        mode (str): 'train', 'evaluate', 'predict' 중 하나.
    """
    print(f"--- Running preprocessing in '{mode}' mode ---")
    df = load_raw_data(raw_data_path) # 1. 원본 데이터 로드

    # 2. 기본 전처리 (타겟 레이블 생성, 날짜 처리)
    df = create_target_label(df)

    # 3. 피처별 전처리 (수치형, 범주형(장르))
    df, lang_onehot_cols = preprocess_language_onehot(df, mode=mode) # 언어 전처리 추가
    df, month_onehot_cols = process_release_date_and_month_onehot(df, mode=mode) # 수정된 함수 호출
    df = preprocess_numerical_features(df, mode=mode)
    df, genre_cols = preprocess_genre_features(df, mode=mode) # 장르 처리 후 생성된 열 이름들도 받음

    # 텍스트 필드 결합 (CLS 추출 전에 수행)
    df = combine_text_fields_for_bert(df)

    # 4. 학습 모드일 때만 CLS 벡터 추출 및 최종 데이터 저장
    if mode == 'train':
        # BERT 토크나이저와 모델을 불러옵니다. (config 파일의 설정값 사용)
        # .eval()은 모델을 평가 모드로 설정 (Dropout 등 비활성화, 벡터 추출 시에는 항상 평가 모드)
        tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)
        bert_model = BertModel.from_pretrained(config.BERT_MODEL_NAME).to(config.DEVICE).eval()
        df = extract_cls_vectors_from_df(df, tokenizer, bert_model, config.DEVICE)

        # 전처리 완료된 데이터를 지정된 경로에 Pickle 파일 형태로 저장합니다.
        # Pickle은 DataFrame을 원래 형태 그대로 저장하고 불러올 수 있어 편리합니다.
        os.makedirs(os.path.dirname(processed_data_path), exist_ok=True) # 저장 폴더 생성
        df.to_pickle(processed_data_path)
        print(f"Preprocessed data saved to {processed_data_path}")
    elif mode == 'predict' or mode == 'evaluate':
        # 예측 또는 평가 모드에서는 보통 이미 전처리된 데이터를 불러와서 사용합니다.
        # 만약 실시간으로 새로운 단일 데이터에 대해 예측해야 한다면,
        # 이 함수를 호출하기 전에 해당 데이터에 대해 CLS 벡터를 미리 추출하거나,
        # 이 함수 내에서 CLS 벡터 추출 로직을 조건부로 실행하도록 수정해야 합니다.
        # 여기서는 학습 데이터 생성에 초점을 맞추고, 예측 시에는 이 함수가 아닌
        # predict.py의 개별 전처리 함수들을 조합해서 사용한다고 가정합니다.
        print("CLS vector extraction skipped in 'predict'/'evaluate' mode for full dataset preprocessing.")
        print("(Assumed to be handled by loading preprocessed data or in predict_single function)")


    print(f"--- Preprocessing '{mode}' mode finished ---")
    # 이제 wide 파트에 사용될 컬럼은 NUMERICAL_FEATURES + genre_cols + month_onehot_cols
    all_wide_cols = config.NUMERICAL_FEATURES + genre_cols + month_onehot_cols + lang_onehot_cols
    return df, all_wide_cols # wide 파트 컬럼명 리스트 반환

if __name__ == '__main__':
    # 이 파일을 파이썬으로 직접 실행할 때 (예: python src/preprocess.py) 아래 코드가 실행됩니다.
    # (주로 테스트나 단독 실행을 위해 사용)
    print("Running preprocess.py as a script...")
    # 학습 모드로 전체 전처리 과정을 실행합니다.
    # config.RAW_DATA_PATH에 원본 데이터 파일이 있거나, 없으면 샘플 데이터가 사용됩니다.
    # 결과는 config.PROCESSED_DATA_PATH에 저장됩니다.
    _, processed_wide_cols = run_preprocessing(
        raw_data_path=config.RAW_DATA_PATH,
        processed_data_path=config.PROCESSED_DATA_PATH,
        mode='train'
    )
    print(f"Preprocessing script finished. All wide columns for model: {processed_wide_cols}")

    # (선택 사항) 전처리된 데이터가 잘 만들어졌는지 간단히 확인해볼 수 있습니다.
    if os.path.exists(config.PROCESSED_DATA_PATH):
        df_processed = pd.read_pickle(config.PROCESSED_DATA_PATH)
        print("\n--- Processed DataFrame Head (first 5 rows) ---")
        print(df_processed.head())
        print("\n--- Processed DataFrame Info (column types, non-null counts) ---")
        df_processed.info()
        if config.CLS_VECTOR_COL in df_processed.columns and len(df_processed) > 0:
            print(f"Length of a CLS vector in the first row: {len(df_processed[config.CLS_VECTOR_COL].iloc[0])}")