from fastapi import FastAPI
from transformers import BertTokenizer, BertModel
from pydantic import BaseModel
from predict import predict_single
import config
import joblib
from model import WideAndDeepModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 모델 및 인코더 로딩
device = config.DEVICE

scaler = joblib.load("../models/standard_scaler.joblib")
mlb = joblib.load("../models/mlb.joblib")
month_ohe = joblib.load("../models/month_onehot_encoder.joblib")
lang_ohe = joblib.load("../models/language_onehot_encoder.joblib")
prodco_mlb = joblib.load("../models/prodco_mlb_encoder.joblib") # 제작사 MLB 로드

tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)
bert_model = BertModel.from_pretrained(config.BERT_MODEL_NAME).to(device).eval()

wide_input_dim = len(config.NUMERICAL_FEATURES) + len(mlb.classes_) + len(month_ohe.categories_[0]) + len(lang_ohe.categories_[0]) + len(prodco_mlb.classes)
deep_input_dim = config.DEEP_INPUT_DIM

model = WideAndDeepModel(
    wide_input_dim=wide_input_dim,
    deep_input_dim=deep_input_dim,
    deep_hidden_dims=config.DEEP_HIDDEN_DIMS,
    dropout_rate=config.DROPOUT_RATE
).to(device)


# 모델 가중치 불러오기
from utils import load_checkpoint
from typing import List
model, _ = load_checkpoint(config.MODEL_WEIGHTS_PATH, model, optimizer=None, device=device)

class MovieInput(BaseModel):
    title: str
    synopsis: str
    keywords: List[str]
    runtime: float
    release_year: int
    release_month: int
    release_day: int
    language: str
    company: List[str]
    genre: List[str]


@app.post("/predict")
def predict(data: MovieInput):
    input_dict = {
        config.ORIG_TITLE_COL: data.title,
        config.ORIG_SYNOPSIS_COL: data.synopsis,
        config.ORIG_KEYWORDS_COL: str(data.keywords),
        config.ORIG_RUNTIME_COL: data.runtime,
        config.ORIG_GENRES_COL: str(data.genre),
        config.ORIG_LANGUAGE_COL: data.language,
        config.ORIG_PRODUCTION_COMPANY_COL: str(data.company),
        config.ORIG_RELEASE_DATE_COL: f"{data.release_year:04d}-{data.release_month:02d}-{data.release_day:02d}"
    }

    #요청이 오는 순간 감시
    print("요청 도착:", input_dict)
    
    prob = predict_single(
        input_dict, model, device,
        tokenizer, bert_model,
        scaler, mlb, month_ohe, lang_ohe, prodco_mlb
    )

    
    print("예측 결과:", prob)

    return {
        "success_probability": round(prob, 4),
        "verdict": "성공 가능성 높음" if prob >= 0.5 else "성공 가능성 낮음"
    }