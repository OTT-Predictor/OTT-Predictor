from fastapi import FastAPI
from transformers import BertTokenizer, BertModel
from pydantic import BaseModel
from predict import predict_single
import config
import joblib
from model import WideAndDeepModel

app = FastAPI()

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

@app.post("/predict")
def predict(data: MovieInput):
    input_dict = data.dict()
    prob = predict_single(
        input_dict, model, device,
        tokenizer, bert_model,
        scaler, mlb, month_ohe, lang_ohe
    )
    return {
        "success_probability": round(prob, 4),
        "verdict": "성공 가능성 높음" if prob >= 0.5 else "성공 가능성 낮음음"
    }