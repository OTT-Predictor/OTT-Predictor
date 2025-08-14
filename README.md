# OTT-Predictor

> 영화/OTT 콘텐츠의 성공 확률을 예측하는 AI 기반 실시간 분석 시스템

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)
![Flutter](https://img.shields.io/badge/Flutter-3.0+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)

## 📋 프로젝트 개요

영화 시장의 규모가 점점 커지고 있고, 그에 비례하여 영화 제작비가 기하급수적으로 증가하나, 흥행 실패로 인한 손실도 지속적으로 증가하고 있습니다. 기존 의사결정은 예측 정확도가 40~45% 수준에 머물러 있어서, 콘텐츠 성공 가능성을 사전에 정량적으로 예측할 수 있는 인공지능 기반의 실시간 분석 시스템이 필요합니다.

본 프로젝트는 영화의 메타데이터(장르, 시놉시스, 키워드 등)를 입력하면 성공 확률을 실시간으로 예측하는 시스템을 구축하는 것을 목표로 합니다.

## 🎯 주요 기능

- **실시간 예측**: 영화 메타데이터를 입력하면 즉시 성공 확률 제공
- **크로스플랫폼**: 웹과 모바일 모두 지원하는 Flutter 앱
- **상위 랭킹**: 성공확률 상위 200위 영화 랭킹 제공
- **직관적 UI**: 사용자 친화적인 반응형 인터페이스

## 🏗️ 시스템 구조
OTT-Predictor/ ├── backend/ # FastAPI 백엔드 │ ├── models/ # AI 모델 파일 │ ├── preprocessing/ # 전처리 파이프라인 │ └── api/ # REST API 엔드포인트 ├── frontend/ # Flutter 프론트엔드 │ ├── lib/ # Dart 소스 코드 │ └── assets/ # UI 리소스 └── data/ # 데이터셋 및 전처리 스크립트


## 🔧 기술 스택

### Backend
- **Framework**: FastAPI
- **ML Libraries**: PyTorch, transformers (BERT)
- **Data Processing**: Pydantic, joblib
- **API Documentation**: Swagger UI

### Frontend
- **Framework**: Flutter/Dart
- **Platforms**: Web, Android, iOS

### AI/ML
- **Model Architecture**: BERT + Wide & Deep
- **Optimization**: W&B Bayesian Sweeps
- **Data Source**: TMDB API

## 📊 모델 구조

### Wide & Deep Architecture
- **Wide Part**: 수치형·범주형 피처를 단일 선형레이어로 처리
- **Deep Part**: BERT CLS 벡터(768차원)를 여러 dense+ReLU+Dropout층으로 변환
- **Output**: 두 파트의 logit을 합친 후 sigmoid로 성공 확률(0~1) 출력

### 입력 피처
- **수치형**: 상영시간, 개봉연도
- **범주형**: 장르, 제작사, 개봉월, 언어
- **텍스트**: 제목+시놉시스+키워드 (BERT 임베딩)

## 📈 데이터셋

- **원본 데이터**: TMDB API에서 567,222건 수집 (1900~2025년)
- **전처리 후**: 11,706건의 고품질 데이터셋
- **성공 기준**: ROI > 1.5, 평점 > 6.5, 평점수 > 1000
- **분할**: 학습/검증 8:2




### 모바일에서 실행
flutter run



