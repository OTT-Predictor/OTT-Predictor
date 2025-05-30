# src/predict.py
import torch
import pandas as pd
import numpy as np
import joblib # 저장된 전처리기(scaler, mlb, ohe)를 불러오기 위해 사용
import os
import ast # 문자열 형태의 리스트를 실제 리스트로 변환하기 위해 사용

# 우리 프로젝트의 다른 파일들을 불러옵니다.
from . import config # 설정값 (config.py)
from .model import WideAndDeepModel # 모델 클래스 (model.py)
# preprocess.py에서 개별 전처리 함수들을 직접 가져와 사용할 수도 있지만,
# 여기서는 예측용 데이터를 간단히 만들기 위해 일부 로직을 포함하거나,
# 또는 preprocess.py에 예측용 단일 데이터 처리 함수를 만들고 호출할 수 있습니다.
# 이번 예시에서는 주요 전처리 단계를 predict.py 내에 간략히 구현합니다.
# BERT 관련 도구 (토크나이저, 모델 - CLS 벡터 추출용)
from transformers import BertTokenizer, BertModel
from .utils import load_checkpoint # 저장된 모델 가중치를 불러오는 함수 (utils.py)

def predict_single(data_dict, model, device, tokenizer, bert_model_for_cls, scaler, mlb, month_ohe, lang_ohe, prodco_mlb):
    """
    하나의 영화 데이터(딕셔너리 형태)에 대한 성공 예측을 수행하는 함수입니다.
    Args:
        data_dict (dict): 예측할 영화의 정보를 담은 딕셔너리.
                          (config.py에 정의된 ORIG_..._COL 이름들을 키로 사용)
        model (WideAndDeepModel): 학습된 모델 객체.
        device (torch.device): 사용할 장치 (GPU 또는 CPU).
        tokenizer (BertTokenizer): BERT 토크나이저.
        bert_model_for_cls (BertModel): CLS 벡터 추출용 BERT 모델.
        scaler (StandardScaler): 학습 시 사용된 수치형 피처 정규화 도구.
        mlb (MultiLabelBinarizer): 학습 시 사용된 장르 인코딩 도구.
        month_ohe (OneHotEncoder): 학습 시 사용된 월 인코딩 도구.
        lang_ohe (OneHotEncoder): 학습 시 사용된 언어 인코딩 도구.
    Returns:
        float: 예측된 성공 확률 (0~1 사이).
    """
    model.eval() # 모델을 "평가 모드"로 설정 (드롭아웃 등 비활성화)

    # --- 1. 입력 데이터(data_dict)를 DataFrame 형태로 변환 (전처리 함수들이 DataFrame 입력을 가정) ---
    # 실제로는 여러 영화를 한 번에 예측할 수도 있으므로, 리스트 안에 딕셔너리를 넣어 DataFrame을 만듭니다.
    df_input = pd.DataFrame([data_dict])

    # --- 2. 데이터 전처리 (학습 때와 동일한 방식으로!) ---
    # 2-1. 날짜 처리 (연, 월, 일 추출)
    if config.ORIG_RELEASE_DATE_COL in df_input.columns:
        df_input[config.ORIG_RELEASE_DATE_COL] = pd.to_datetime(df_input[config.ORIG_RELEASE_DATE_COL], errors='coerce')
        df_input['release_year'] = df_input[config.ORIG_RELEASE_DATE_COL].dt.year
        df_input['release_month_orig'] = df_input[config.ORIG_RELEASE_DATE_COL].dt.month # OHE용 월
        df_input['release_day'] = df_input[config.ORIG_RELEASE_DATE_COL].dt.day
        for col in ['release_year', 'release_month_orig', 'release_day']:
            df_input[col] = df_input[col].fillna(-1).astype(int) # 결측치 처리
    else: # 만약 날짜 정보가 없다면, 해당 컬럼들을 -1로 채워줍니다. (모델 입력 차원 맞추기 위해)
        for col_name in ['release_year', 'release_month_orig', 'release_day']:
             if col_name not in df_input.columns: df_input[col_name] = -1


    # 2-2. 수치형 피처 정규화 (학습 때 사용한 scaler로 transform!)
    # config.NUMERICAL_FEATURES 에 정의된 순서대로 값이 있어야 함
    numerical_data_for_scaling = df_input[config.NUMERICAL_FEATURES].fillna(0).values # 결측치 0으로
    scaled_numerical_values = scaler.transform(numerical_data_for_scaling)
    df_scaled_numerical = pd.DataFrame(scaled_numerical_values, columns=config.NUMERICAL_FEATURES, index=df_input.index)

    # 2-3. 장르 피처 멀티-핫 인코딩 (학습 때 사용한 mlb로 transform!)
    genres_str = df_input[config.ORIG_GENRES_COL].fillna('[]').iloc[0] # 단일 데이터이므로 iloc[0]
    try: genre_list = ast.literal_eval(genres_str)
    except: genre_list = []
    genre_list_cleaned = [[str(g).strip() for g in genre_list if str(g).strip()]] # 이중 리스트 형태
    
    encoded_genres_arr = mlb.transform(genre_list_cleaned)
    genre_cols = [f"{config.GENRE_MLB_PREFIX}{cls}" for cls in mlb.classes_]
    df_genre_encoded = pd.DataFrame(encoded_genres_arr, columns=genre_cols, index=df_input.index)

    # 2-4. 월 피처 원-핫 인코딩 (학습 때 사용한 month_ohe로 transform!)
    month_data_for_ohe = df_input[['release_month_orig']]
    month_encoded_arr = month_ohe.transform(month_data_for_ohe)
    month_cols = [f"{config.MONTH_OHE_PREFIX}{int(cat)}" for cat in month_ohe.categories_[0]]
    df_month_encoded = pd.DataFrame(month_encoded_arr, columns=month_cols, index=df_input.index)

    # 2-5. 언어 피처 원-핫 인코딩 (학습 때 사용한 lang_ohe로 transform!)
    lang_data_for_ohe = df_input[[config.ORIG_LANGUAGE_COL]].fillna('unknown').astype(str)
    lang_encoded_arr = lang_ohe.transform(lang_data_for_ohe)
    lang_cols = [f"{config.LANGUAGE_OHE_PREFIX}{cat}" for cat in lang_ohe.categories_[0]]
    df_lang_encoded = pd.DataFrame(lang_encoded_arr, columns=lang_cols, index=df_input.index)

        # --- 2-6. 제작사 피처 멀티-핫 인코딩 (학습 때 사용한 prodco_mlb로 transform) ---
    if config.ORIG_PRODUCTION_COMPANY_COL in df_input.columns:
        prodco_str = df_input[config.ORIG_PRODUCTION_COMPANY_COL].fillna('[]').iloc[0]
        try: prodco_list = ast.literal_eval(prodco_str)
        except: prodco_list = []
        prodco_list_cleaned = [[str(c).strip() for c in prodco_list if str(c).strip()]] # 이중 리스트 형태
        
        # prodco_mlb 객체 사용
        prodco_encoded_arr = prodco_mlb.transform(prodco_list_cleaned)
        prodco_cols = [f"{config.PRODUCTION_COMPANY_MLB_PREFIX}{cls}" for cls in prodco_mlb.classes_]
        df_prodco_encoded = pd.DataFrame(prodco_encoded_arr, columns=prodco_cols, index=df_input.index)
    else: # 제작사 컬럼이 입력에 없다면, 모든 제작사 OHE 컬럼을 0으로 채움
        if os.path.exists(config.PRODCO_MLB_PATH):
             prodco_mlb_loaded = joblib.load(config.PRODCO_MLB_PATH) # 컬럼명만 알기 위해 로드
             prodco_cols = [f"{config.PRODUCTION_COMPANY_MLB_PREFIX}{cls}" for cls in prodco_mlb_loaded.classes_]
             df_prodco_encoded = pd.DataFrame(0, index=df_input.index, columns=prodco_cols)
        else:
             prodco_cols = [] # MLB 파일도 없다면 제작사 컬럼 없음
             df_prodco_encoded = pd.DataFrame(index=df_input.index) # 빈 DataFrame
        print("Warning: Production company column not found in input data. Using zero vector for prodco features.")

    # 2-7. 텍스트 필드 결합 (title, synopsis, keywords)
    title_text = str(df_input[config.ORIG_TITLE_COL].fillna('').iloc[0])
    synopsis_text = str(df_input[config.ORIG_SYNOPSIS_COL].fillna('').iloc[0])
    keywords_input_str = str(df_input[config.ORIG_KEYWORDS_COL].fillna('[]').iloc[0])
    try: keywords_list_parsed = ast.literal_eval(keywords_input_str)
    except: keywords_list_parsed = []
    keywords_text = ", ".join([str(k).strip() for k in keywords_list_parsed if str(k).strip()])
    
    sep_token_text = " [SEP] "
    combined_text = title_text + sep_token_text + synopsis_text + sep_token_text + keywords_text
    
    # 2-7. CLS 벡터 추출 (결합된 텍스트 사용)
    # get_cls_vector_batch 함수는 preprocess.py 에 정의되어 있다고 가정하고 사용
    # 또는 해당 함수 로직을 여기에 직접 구현
    from .preprocess import get_cls_vector_batch # preprocess.py에서 함수 가져오기
    cls_vector_arr = get_cls_vector_batch([combined_text], tokenizer, bert_model_for_cls, device, config.BERT_MAX_LENGTH)


    # --- 3. 모델 입력 형태로 변환 ---
    # Wide 파트 입력: 정규화된 수치형 + 인코딩된 장르 + 인코딩된 월 + 인코딩된 언어
    # 순서가 매우 중요! Dataset 만들 때와 동일한 순서여야 함.
    df_wide_features = pd.concat([df_scaled_numerical, df_genre_encoded, df_month_encoded, df_lang_encoded, df_prodco_encoded], axis=1)
    wide_input_features = df_wide_features.values.astype(np.float32) # NumPy 배열로

    # PyTorch 텐서로 변환하고 지정된 장치(GPU/CPU)로 옮깁니다.
    wide_input_tensor = torch.tensor(wide_input_features, dtype=torch.float).to(device)
    deep_input_tensor = torch.tensor(cls_vector_arr, dtype=torch.float).to(device) # CLS 벡터는 이미 (1, 768) 형태

    # 모델의 forward 함수가 받을 딕셔너리 형태로 만듭니다.
    model_input_dict = {
        'wide_input': wide_input_tensor,
        'deep_input': deep_input_tensor
    }

    # --- 4. 예측 수행 ---
    with torch.no_grad(): # 기울기 계산 비활성화 (예측 시에는 필요 없음)
        prediction_proba = model(model_input_dict) # 모델의 forward 함수 호출

    return prediction_proba.item() # 예측된 확률값(스칼라) 반환

def main_predict():
    """예측 스크립트의 메인 실행 함수입니다."""
    device = config.DEVICE # 사용할 장치 (config.py 에서 설정)
    print(f"Using device: {device} for prediction.")

    # --- 1. 필요한 전처리기, 토크나이저, 모델 등 로드 ---
    print("Loading preprocessors and models...")
    try:
        scaler = joblib.load(config.SCALER_PATH)
        mlb = joblib.load(config.MLB_PATH)
        month_ohe = joblib.load(config.MONTH_OHE_PATH)
        lang_ohe = joblib.load(config.LANGUAGE_OHE_PATH)
        prodco_mlb = joblib.load(config.PRODCO_MLB_PATH) # 제작사 MLB 로드
    except FileNotFoundError as e:
        print(f"Error: Preprocessor file not found. {e}")
        print("Please run training first to generate these preprocessor files.")
        return

    tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    # CLS 벡터 추출용 BERT 모델 (학습에 사용된 것과 동일해야 함, 평가 모드로)
    bert_model_for_cls = BertModel.from_pretrained(config.BERT_MODEL_NAME).to(device).eval()

    # Wide 파트 입력 차원 결정 (저장된 전처리기들로부터)
    actual_wide_input_dim = len(config.NUMERICAL_FEATURES) + \
                            len(mlb.classes_) + \
                            len(month_ohe.categories_[0]) + \
                            len(lang_ohe.categories_[0]) + \
                            len(prodco_mlb.classes_) # 제작사 컬럼 수 추가)
    print(f"Determined WIDE_INPUT_DIM for the model: {actual_wide_input_dim}")

    # 우리 모델(WideAndDeepModel) 객체 생성
    model_instance = WideAndDeepModel(
        wide_input_dim=actual_wide_input_dim,
        deep_input_dim=config.DEEP_INPUT_DIM,
        deep_hidden_dims=config.DEEP_HIDDEN_DIMS,
        dropout_rate=config.DROPOUT_RATE # 예측 시에는 드롭아웃 비율을 0으로 하거나, model.eval()이 처리
    ).to(device)

    # 학습된 모델 가중치 불러오기 (utils.py의 함수 사용)
    if not os.path.exists(config.MODEL_WEIGHTS_PATH):
        print(f"Error: Model weights not found at {config.MODEL_WEIGHTS_PATH}. Please train the model first.")
        return
    # load_checkpoint는 모델 객체에 직접 가중치를 로드합니다. 옵티마이저 상태는 예측 시 필요 없음.
    model_instance, _ = load_checkpoint(config.MODEL_WEIGHTS_PATH, model_instance, optimizer=None, device=device)
    print("Trained model weights loaded successfully.")


    # --- 2. 예측할 가상의 영화 데이터 준비 ---
    # (config.py에 정의된 ORIG_..._COL 이름들을 키로 사용해야 함)
    virtual_movie_to_predict = {
        config.ORIG_TITLE_COL: '타이타닉',
        config.ORIG_SYNOPSIS_COL: '1912년 4월 귀족 가문의 딸인 로즈는 타이타닉호에 승선한다. 사랑하지도 않는 남자와 결혼해야 한다는 사실에 절망한 로즈는 자살을 시도하지만, 마침 가난한 화가인 잭이 구해주게 되고 둘은 서로 사랑에 빠진다. 미래를 함께 약속하며 행복한 시간을 보내는 것도 잠시, 로즈와 잭은 예상치 못한 난관과 마주하게 된다.',
        config.ORIG_KEYWORDS_COL: "['shipwerck', 'romance', 'tragedy', 'iceberg', 'ocean']", # 문자열 리스트 형태
        config.ORIG_RUNTIME_COL: 115,
        config.ORIG_RELEASE_DATE_COL: '1980-08-01', # 미래 날짜
        config.ORIG_GENRES_COL: "['모험']", # 문자열 리스트 형태
        config.ORIG_LANGUAGE_COL: 'en', # 한국어
        config.ORIG_PRODUCTION_COMPANY_COL: '20th Century Fox' # 제작사
    }
    # 만약 release_date 대신 year, month, day를 직접 넣는다면 아래처럼
    # 'release_year': 2025,
    # 'release_month_orig': 12, # OHE 전 원래 월 값
    # 'release_day': 20,
    #virtual_movie_to_predict = {
    #    config.ORIG_TITLE_COL: '미키 17',
    #    config.ORIG_SYNOPSIS_COL: '친구 티모와 함께 차린 마카롱 가게가 쫄딱 망해 거액의 빚을 지고 못 갚으면 죽이겠다는 사채업자를 피해 지구를 떠나야 하는 미키. 기술이 없는 그는, 정치인 마셜의 얼음행성 개척단에서 위험한 일을 도맡고, 죽으면 다시 프린트되는 익스펜더블로 지원한다. 4년의 항해와 얼음행성 니플하임에 도착한 뒤에도 늘 미키를 지켜준 여자친구 나샤. 그와 함께, 미키는 반복되는 죽음과 출력의 사이클에도 익숙해진다. 그러나 미키 17이 얼음행성의 생명체인 크리퍼와 만난 후 죽을 위기에서 돌아와 보니 이미 미키 18이 프린트되어 있다. 행성 당 1명만 허용된 익스펜더블이 둘이 된 멀티플 상황. 둘 중 하나는 죽어야 하는 현실 속에 걷잡을 수 없는 사건이 기다리고 있었으니…',
    #    config.ORIG_KEYWORDS_COL: "['based on novel or book', 'dark comedy', 'space travel', 'space colony', 'alien planet', 'creature', 'space adventure', 'human cloning', 'spaceship', 'space sci-fi', 'black comedy']", # 문자열 리스트 형태
    #    config.ORIG_REVENUE_COL: 127337252, # 단위: 억 (가정)
    #    config.ORIG_BUDGET_COL: 118000000,  # 단위: 억 (가정)
    #    config.ORIG_RATING_COL: 8.2,  
    #    config.ORIG_RUNTIME_COL: 137,
    #    config.ORIG_RELEASE_DATE_COL: '2026-02-28', # 미래 날짜
    #    config.ORIG_GENRES_COL: "['SF', '코미디', '모험']", # 문자열 리스트 형태
    #    config.ORIG_LANGUAGE_COL: 'en' 
    #}
    print("\n--- Predicting for Virtual Movie ---")
    print(f"Input data: {virtual_movie_to_predict}")

    # --- 3. 예측 실행 ---
    predicted_probability = predict_single(
        virtual_movie_to_predict, model_instance, device,
        tokenizer, bert_model_for_cls,
        scaler, mlb, month_ohe, lang_ohe, prodco_mlb
    )

    # --- 4. 결과 출력 ---
    # config.PREDICTION_THRESHOLD (예: 0.5) 기준으로 성공/실패 판단
    predicted_label = 1 if predicted_probability >= config.PREDICTION_THRESHOLD else 0

    print(f"\nPredicted Success Probability: {predicted_probability:.4f}")
    print(f"Predicted Label (Threshold {config.PREDICTION_THRESHOLD}): {'성공' if predicted_label == 1 else '실패'}")

if __name__ == '__main__':
    # 이 파일을 직접 실행할 때 (python -m src.predict 또는 python src/predict.py)
    # main_predict() 함수가 호출되어 예측 과정이 시작됩니다.
    main_predict()