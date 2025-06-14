# src/train.py
import torch
import torch.optim as optim # 옵티마이저(Adam 등)를 사용하기 위해 불러옴
import torch.nn as nn     # 손실 함수(BCELoss 등) 및 모델 기본 클래스(nn.Module) 사용
# DataLoader는 데이터를 배치 단위로 묶어주고, random_split은 데이터를 나누는 데 사용
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
import pandas as pd # WIDE_INPUT_DIM 결정 시 임시로 DataFrame 정보를 볼 때 사용될 수 있음
from tqdm import tqdm # 학습 진행 상황을 보여주는 막대바 라이브러리
from sklearn.model_selection import train_test_split # sklearn 임포트 추가
from torch.optim.lr_scheduler import ReduceLROnPlateau # 학습률 스케줄러 예시

# W&B 라이브러리 임포트
import wandb

# 우리 프로젝트의 다른 파일들을 불러옵니다.
from . import config # 설정값 (config.py)
from .dataset import MovieSuccessDataset # 데이터셋 클래스 (dataset.py)
from .model import WideAndDeepModel    # 모델 클래스 (model.py)
from .preprocess import run_preprocessing # 데이터 전처리 함수 (preprocess.py)
# 유용한 함수들 (utils.py)
from .utils import seed_everything, save_checkpoint, load_checkpoint, calculate_metrics
import joblib # MLB, OHE 로드하여 컬럼명 얻기 위해 추가

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """한 에포크(전체 데이터 1회 학습) 동안 모델을 학습시키는 함수입니다."""
    model.train() # 모델을 "학습 모드"로 설정 (드롭아웃 등 활성화)
    running_loss = 0.0 # 현재 에포크의 총 손실을 기록할 변수
    # tqdm: dataloader에서 데이터를 가져올 때 진행 상황 막대바로 표시
    for batch_data in tqdm(dataloader, desc=f"Training Epoch"):
        # 1. 데이터 준비: 입력과 타겟(정답)을 지정된 장치(GPU 또는 CPU)로 옮깁니다.
        model_inputs = { # 모델의 forward 함수가 받을 딕셔너리 형태
            'wide_input': batch_data['wide_input'].to(device),
            'deep_input': batch_data['deep_input'].to(device)
        }
        targets = batch_data['target'].to(device)

        # 2. 기울기 초기화: 이전 배치의 기울기가 남아있지 않도록 0으로 만듭니다. (매우 중요!)
        optimizer.zero_grad()

        # 3. 순전파 (Forward Pass): 모델에 입력을 넣어 예측값을 계산합니다.
        outputs = model(model_inputs) # 모델의 forward 함수 호출

        # 4. 손실 계산: 예측값과 실제 정답을 비교하여 손실(오차)을 계산합니다.
        loss = criterion(outputs, targets)

        # 5. 역전파 (Backward Pass): 계산된 손실을 바탕으로 모델의 각 부분(가중치)이
        #    손실에 얼마나 영향을 미쳤는지(기울기)를 계산합니다.
        loss.backward()

        # 6. 파라미터 업데이트: 계산된 기울기를 사용하여 옵티마이저가 모델의 가중치를 수정합니다.
        #    (손실을 줄이는 방향으로)
        optimizer.step()

        running_loss += loss.item() # 현재 배치의 손실값을 더해줍니다. (.item()은 텐서에서 숫자 값만 가져옴)
    
    # 현재 에포크의 평균 손실을 반환합니다.
    return running_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device, threshold=config.PREDICTION_THRESHOLD):
    """모델의 성능을 평가하는 함수입니다 (검증 또는 테스트용)."""
    model.eval() # 모델을 "평가 모드"로 설정 (드롭아웃 등 비활성화, 예측 결과 일관성 유지)
    total_loss = 0.0 # 총 손실
    all_targets = [] # 모든 실제 정답들을 모을 리스트
    all_predictions_proba = [] # 모든 예측 확률값들을 모을 리스트

    with torch.no_grad(): # 평가 시에는 기울기 계산이 필요 없으므로, 계산 과정을 비활성화 (메모리 절약, 속도 향상)
        for batch_data in tqdm(dataloader, desc="Evaluating"):
            model_inputs = {
                'wide_input': batch_data['wide_input'].to(device),
                'deep_input': batch_data['deep_input'].to(device)
            }
            targets = batch_data['target'].to(device)

            outputs = model(model_inputs) # 예측 확률값 (0~1 사이)
            loss = criterion(outputs, targets) # 손실 계산
            total_loss += loss.item()

            all_targets.extend(targets.cpu().numpy()) # CPU로 옮기고 NumPy 배열로 변환하여 리스트에 추가
            all_predictions_proba.extend(outputs.cpu().numpy())

    avg_loss = total_loss / len(dataloader) # 평균 손실
    # utils.py의 calculate_metrics 함수를 사용해 여러 평가 지표 계산
    metrics = calculate_metrics(np.array(all_targets), np.array(all_predictions_proba), threshold)
    return avg_loss, metrics # 평균 손실과 다른 평가 지표들 반환

def parse_deep_hidden_dims(dims_str):
    """문자열 형태의 은닉층 크기(예: "256,128")를 정수 리스트로 변환"""
    return [int(d.strip()) for d in dims_str.split(',')]

def main():
    """메인 학습 파이프라인을 실행하는 함수입니다."""
    # --- W&B 초기화 ---
    # 직접 실행 시 사용할 기본 하이퍼파라미터 (config.py에서 가져옴)
    default_wandb_config = {
        "learning_rate": config.LEARNING_RATE,
        "batch_size": config.BATCH_SIZE,
        "num_epochs": config.NUM_EPOCHS,
        "bert_model_name": config.BERT_MODEL_NAME,
        # deep_hidden_dims_choice는 문자열이어야 하므로, config.DEEP_HIDDEN_DIMS를 문자열로 변환
        "deep_hidden_dims_choice": ",".join(map(str, config.DEEP_HIDDEN_DIMS)),
        "dropout_rate": config.DROPOUT_RATE,
        "random_seed": config.RANDOM_SEED,
    }
    run = wandb.init(project="movie-success-predictor",
                     config=default_wandb_config) # entity는 필요시 설정
    # Sweep 실행 시에는 wandb.init()이 호출될 때 Sweep 컨트롤러가 전달한 값으로 run.config가 덮어쓰여짐.

    # 이제 run.config.parameter_name 형태로 안전하게 접근 가능
    current_learning_rate = run.config.learning_rate
    current_batch_size = run.config.batch_size
    current_num_epochs = run.config.num_epochs
    current_deep_hidden_dims_str = run.config.deep_hidden_dims_choice
    current_dropout_rate = run.config.dropout_rate
    
    if run: # run 객체가 성공적으로 생성되었는지 확인
        print(f"W&B Run Page: {run.get_url()}") # <--- URL 직접 출력!
    else:
        print("Failed to initialize W&B run.")
    # --------------------

    seed_everything(config.RANDOM_SEED) # 결과 재현을 위해 랜덤 시드 고정
    device = config.DEVICE # 사용할 장치 (GPU 또는 CPU)
    print(f"Using device: {device}")
    print(f"W&B Run Config: {run.config}") # Sweep에서 받은 설정값 확인


    # --- 1. 데이터 전처리 (필요한 경우에만 실행) ---
    # 만약 전처리된 데이터 파일이 없다면, preprocess.py의 run_preprocessing 함수를 실행합니다.
    if not os.path.exists(config.PROCESSED_DATA_PATH):
        print(f"Processed data not found at {config.PROCESSED_DATA_PATH}. Running preprocessing...")
        # run_preprocessing은 전처리된 DataFrame과 Wide 파트에 사용될 컬럼명 리스트를 반환합니다.
        _, wide_feature_columns_from_preprocess = run_preprocessing(
            raw_data_path=config.RAW_DATA_PATH,
            processed_data_path=config.PROCESSED_DATA_PATH,
            mode='train' # 학습용 데이터 전처리 모드
        )
        # 이 wide_feature_columns_from_preprocess 리스트의 길이가 WIDE_INPUT_DIM이 됩니다.
        actual_wide_input_dim = len(wide_feature_columns_from_preprocess)
    else: # 전처리된 데이터 파일이 이미 있다면
        print(f"Using existing processed data from {config.PROCESSED_DATA_PATH}")
        # WIDE_INPUT_DIM을 결정하기 위해, 저장된 전처리기(MLB, OHE)를 로드하여 컬럼 수를 계산합니다.
        # (Dataset 클래스 내부에서도 이 로직이 있지만, 모델 생성 전에 알아야 하므로 여기서도 수행)
        numerical_cols_count = len(config.NUMERICAL_FEATURES)
        genre_cols_count = 0
        if os.path.exists(config.MLB_PATH):
            mlb = joblib.load(config.MLB_PATH)
            genre_cols_count = len(mlb.classes_)
        
        month_ohe_cols_count = 0
        if os.path.exists(config.MONTH_OHE_PATH):
            month_ohe = joblib.load(config.MONTH_OHE_PATH)
            month_ohe_cols_count = len(month_ohe.categories_[0])

        lang_ohe_cols_count = 0
        if os.path.exists(config.LANGUAGE_OHE_PATH):
            lang_ohe = joblib.load(config.LANGUAGE_OHE_PATH)
            lang_ohe_cols_count = len(lang_ohe.categories_[0])

        prodco_cols_count = 0
        # 제작사 MLB 로드하여 컬럼 수 계산 추가
        if os.path.exists(config.PRODCO_MLB_PATH):
            prodco_mlb = joblib.load(config.PRODCO_MLB_PATH)
            prodco_cols_count = len(prodco_mlb.classes_)
        
        actual_wide_input_dim = numerical_cols_count + genre_cols_count + month_ohe_cols_count + lang_ohe_cols_count + prodco_cols_count

    print(f"Determined WIDE_INPUT_DIM for the model: {actual_wide_input_dim}")

    # --- 2. 데이터셋 및 데이터로더 생성 ---
    # MovieSuccessDataset 객체를 만듭니다. (전처리된 데이터 파일 경로 사용)
    full_dataset = MovieSuccessDataset(data_path=config.PROCESSED_DATA_PATH, mode='train')

    # full_dataset.targets_data 는 Dataset 클래스에서 로드한 타겟 레이블 NumPy 배열이라고 가정
    # (만약 Dataset 클래스에 targets_data 속성이 없다면, full_dataset.dataframe[config.TARGET_COL].values 사용)
    targets_for_stratify = full_dataset.targets_data 

    # 전체 데이터셋의 인덱스를 만듭니다.
    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.2, # 검증 데이터 비율 (config 파일로 옮겨도 좋음)
        stratify=targets_for_stratify, # 이 부분을 통해 계층적 분할 수행
        random_state=config.RANDOM_SEED # 재현성을 위해 random_state 설정
    )
    # 분할된 인덱스를 사용하여 Subset을 만듭니다.
    from torch.utils.data import Subset # Subset 임포트 추가
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # DataLoader: Dataset에서 데이터를 배치 크기만큼 가져오고, 섞어주는(shuffle) 등의 역할을 합니다.
    train_loader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=current_batch_size, shuffle=False, num_workers=0) # 검증 시에는 섞지 않음

    # --- 3. 모델, 손실함수, 옵티마이저 정의 ---
    # WideAndDeepModel 객체를 만듭니다. (config 파일의 설정값과 위에서 결정한 actual_wide_input_dim 사용)
    # Sweep에서 받은 deep_hidden_dims_choice 파싱
    parsed_deep_hidden_dims = parse_deep_hidden_dims(current_deep_hidden_dims_str)

    model = WideAndDeepModel(
        wide_input_dim=actual_wide_input_dim, # 실제 Wide 입력 피처 수
        deep_input_dim=config.DEEP_INPUT_DIM,
        deep_hidden_dims=parsed_deep_hidden_dims, # 파싱된 은닉층 크기 리스트
        dropout_rate=current_dropout_rate # sweep에서 받은 드롭아웃 비율
    ).to(device) # 모델을 지정된 장치(GPU 또는 CPU)로 옮깁니다.

    # --- W&B 모델 추적 (선택 사항, 모델 구조 및 그래디언트 시각화) ---
    wandb.watch(model, log="all", log_freq=100) # log="all"은 가중치와 그래디언트 모두, log_freq는 N 배치마다
    # -------------------------------------------------------------

    criterion = nn.BCELoss() # 손실 함수: 이진 교차 엔트로피 (성공/실패 예측 문제에 적합)
    optimizer = optim.Adam(model.parameters(), lr=current_learning_rate) # 옵티마이저: Adam 사용
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3) # 학습률을 조정하는 스케쥴러

    # (선택 사항) 만약 이전에 저장된 모델 가중치가 있다면 불러와서 학습을 이어갈 수 있습니다.
    # if os.path.exists(config.MODEL_WEIGHTS_PATH):
    #     model, optimizer = load_checkpoint(config.MODEL_WEIGHTS_PATH, model, optimizer, device)
    #     print("Resuming training from checkpoint.")

    # --- 4. 학습 루프 실행 ---
    print("\n--- Starting Training ---")
    # 조기 종료는 Sweep 실행 시간을 줄이는 데 매우 중요
    best_val_loss_for_early_stop = float('inf')
    epochs_no_improve = 0
    patience_early_stop = 15 # 조기 종료 patience (Sweep 시에는 짧게 설정하는 것이 좋음)
    best_val_f1 = 0.0 # 가장 좋았던 검증 F1 점수를 기록할 변수 (또는 다른 지표 사용 가능)
    history = { # 학습 과정을 기록할 딕셔너리
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_roc_auc': []
    }

    num_epochs_to_run = config.NUM_EPOCHS # 기본 에포크 수 (YAML에서 설정 안 했다면)
    if hasattr(run.config, 'num_epochs'): # YAML에 num_epochs가 정의되어 있다면 그 값 사용
        num_epochs_to_run = run.config.num_epochs
        
    for epoch in range(num_epochs_to_run): # 정해진 에포크 수만큼 반복
        # 한 에포크 학습 실행
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        # 한 에포크 학습 후 검증 데이터로 성능 평가
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        # history 딕셔너리에 현재 에포크 결과 기록
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1_score'])
        history['val_roc_auc'].append(val_metrics['roc_auc'])
        
        # W&B 로그 기록 (val_f1을 포함하여 metric.name과 일치하는 키가 있어야 함)
        log_data = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_metrics['accuracy'],
            "val_precision": val_metrics['precision'],
            "val_recall": val_metrics['recall'],
            "val_f1": val_metrics['f1_score'], # YAML의 metric.name과 일치!
            "val_roc_auc": val_metrics['roc_auc'],
            "learning_rate": optimizer.param_groups[0]['lr']
        }
        wandb.log(log_data)

        scheduler.step(val_loss)
        
        # 현재 에포크의 학습 결과 출력
        print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}] | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Accuracy: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1_score']:.4f} | "
              f"Val Precision: {val_metrics['precision']:.4f} | " # <--- 정밀도 출력 추가
              f"Val Recall: {val_metrics['recall']:.4f} | "       # <--- 재현율 출력 추가
              f"Val ROC_AUC: {val_metrics['roc_auc']:.4f}")

        # 현재 검증 F1 점수가 이전에 기록된 최고 F1 점수보다 좋으면
        current_val_f1 = val_metrics['f1_score']
        if current_val_f1 > best_val_f1:
            best_val_f1 = current_val_f1 # 최고 점수 업데이트
            # 모델 가중치와 옵티마이저 상태를 파일로 저장 (utils.py의 함수 사용)
            save_checkpoint(model.state_dict(), optimizer.state_dict(), config.MODEL_WEIGHTS_PATH)
            print(f"Best F1 score improved to {best_val_f1:.4f}. Model checkpoint saved to {config.MODEL_WEIGHTS_PATH}")

        # 조기 종료 로직
        if val_loss < best_val_loss_for_early_stop:
            best_val_loss_for_early_stop = val_loss
            epochs_no_improve = 0
            # Sweep 중에는 매번 모델을 저장할 필요는 없을 수 있음 (최종 모델은 Sweep 종료 후 선택)
            # save_checkpoint(model.state_dict(), None, os.path.join(run.dir, "best_model.pth")) # W&B run 디렉토리에 저장
        else:
            epochs_no_improve +=1
        
        if epochs_no_improve >= patience_early_stop:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print("--- Training Finished ---")
    # 학습 완료 후 history를 파일로 저장 (예: pickle)
    import pickle
    with open(os.path.join(config.MODEL_DIR, 'training_history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    print("Training history saved.")

    # --- W&B 실험 종료 ---
    run.finish()
    # --------------------

if __name__ == '__main__':
    # 이 파일을 직접 실행할 때 (python -m src.train 또는 python src/train.py)
    # main() 함수가 호출되어 전체 학습 과정이 시작됩니다.
    main()