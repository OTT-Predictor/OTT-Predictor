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

tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)
bert_model = BertModel.from_pretrained(config.BERT_MODEL_NAME).to(device).eval()

wide_input_dim = len(config.NUMERICAL_FEATURES) + len(mlb.classes_) + len(month_ohe.categories_[0]) + len(lang_ohe.categories_[0])
deep_input_dim = config.DEEP_INPUT_DIM

model = WideAndDeepModel(
    wide_input_dim=wide_input_dim,
    deep_input_dim=deep_input_dim,
    deep_hidden_dims=config.DEEP_HIDDEN_DIMS,
    dropout_rate=config.DROPOUT_RATE
).to(device)


# 모델 가중치 불러오기
from utils import load_checkpoint
model, _ = load_checkpoint(config.MODEL_WEIGHTS_PATH, model, optimizer=None, device=device)

class MovieInput(BaseModel):
    title: str
    keywords: list
    genre: list
    runtime: float
    budget: float
    release_year: int
    release_month: int
    language: str
    company: str
    synopsis: str

@app.post("/predict")
def predict(data: MovieInput):
    input_dict = data.dict()

    
    input_dict[config.ORIG_RUNTIME_COL] = input_dict.pop("runtime")
    input_dict[config.ORIG_GENRES_COL] = input_dict.pop("genre")
    input_dict[config.ORIG_LANGUAGE_COL] = input_dict.pop("language")
    input_dict[config.ORIG_TITLE_COL] = input_dict.pop("title")
    input_dict[config.ORIG_SYNOPSIS_COL] = input_dict.pop("synopsis")  # "시놉시스"
    input_dict[config.ORIG_KEYWORDS_COL] = input_dict.pop("keywords")  # "키워드"


    #요청이 오는 순간 감시
    print("요청 도착:", input_dict)
    
    prob = predict_single(
        input_dict, model, device,
        tokenizer, bert_model,
        scaler, mlb, month_ohe, lang_ohe
    )

    
    print("예측 결과:", prob)

    return {
        "success_probability": round(prob, 4),
        "verdict": "성공 가능성 높음" if prob >= 0.5 else "성공 가능성 낮음"
    }