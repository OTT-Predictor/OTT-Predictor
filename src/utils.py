# src/utils.py
import torch  # PyTorch 라이브러리 (모델 저장/로드, 시드 고정 등에 사용)
import random # 파이썬 기본 랜덤 숫자 생성기
import numpy as np # NumPy 라이브러리 (숫자 배열 다루기, 랜덤 시드 고정 등에 사용)
import os     # 운영체제 기능 사용 (파일 경로, 폴더 생성 등)
# scikit-learn 라이브러리에서 모델 성능 평가 지표 계산 함수들을 불러옵니다.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def seed_everything(seed):
    """
    재현성을 위해 랜덤 시드를 고정하는 함수입니다.
    이 함수를 호출하면, 다음에 프로그램을 실행해도 랜덤으로 결정되는 부분들이
    항상 똑같은 순서와 값으로 나오게 되어 실험 결과를 비교하기 용이해집니다.
    Args:
        seed (int): 고정할 시드 값 (아무 숫자나 괜찮습니다).
    """
    random.seed(seed) # 파이썬 내장 random 모듈의 시드 고정
    os.environ['PYTHONHASHSEED'] = str(seed) # 파이썬 해시 시드 고정
    np.random.seed(seed) # NumPy의 랜덤 시드 고정
    torch.manual_seed(seed) # PyTorch의 CPU 연산 시드 고정
    if torch.cuda.is_available(): # 만약 GPU를 사용한다면
        torch.cuda.manual_seed(seed) # 현재 GPU의 시드 고정
        torch.cuda.manual_seed_all(seed) # 모든 GPU의 시드 고정 (여러 GPU 사용 시)
        # 아래 두 줄은 GPU 연산 시 결과 재현성을 위한 추가 설정입니다.
        # True로 하면 결과는 항상 같지만, 학습 속도가 약간 느려질 수 있습니다.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}") # 시드가 고정되었음을 알려줍니다.

def save_checkpoint(model_state, optimizer_state, filepath):
    """
    모델의 학습 상태(가중치)와 옵티마이저의 상태를 파일로 저장하는 함수입니다.
    학습 중간에 저장해두면, 나중에 이어서 학습하거나 학습된 모델을 사용할 수 있습니다.
    Args:
        model_state (dict): model.state_dict()로 얻은 모델 가중치 정보.
        optimizer_state (dict): optimizer.state_dict()로 얻은 옵티마이저 상태 정보. (필요 없다면 None 전달 가능)
        filepath (str): 저장할 파일 경로 및 이름.
    """
    state = {'model_state_dict': model_state} # 모델 가중치를 state 딕셔너리에 저장
    if optimizer_state: # 만약 옵티마이저 상태도 저장하고 싶다면
        state['optimizer_state_dict'] = optimizer_state # 옵티마이저 상태도 추가
    
    # 파일 저장 전에 폴더가 없으면 만들어줍니다. (예: models 폴더)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath) # PyTorch를 이용해 state 딕셔너리를 파일로 저장
    print(f"Checkpoint saved to {filepath}") # 저장 완료 메시지 출력

def load_checkpoint(filepath, model, optimizer=None, device='cpu'):
    """
    저장된 체크포인트 파일로부터 모델 가중치와 옵티마이저 상태를 불러오는 함수입니다.
    Args:
        filepath (str): 불러올 체크포인트 파일 경로.
        model (torch.nn.Module): 가중치를 로드할 모델 객체.
        optimizer (torch.optim.Optimizer, optional): 상태를 로드할 옵티마이저 객체. 기본값은 None.
        device (str, optional): 모델을 로드할 장치 ('cpu' 또는 'cuda'). 기본값은 'cpu'.
    Returns:
        model, optimizer: 가중치/상태가 로드된 모델과 옵티마이저.
    """
    if not os.path.exists(filepath): # 파일이 존재하지 않으면
        print(f"Checkpoint file not found: {filepath}")
        return model, optimizer # 원래 모델과 옵티마이저를 그대로 반환 (또는 오류 처리)

    # map_location=device 는 GPU에서 저장된 모델을 CPU에서 불러오거나 그 반대의 경우를 처리해줍니다.
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict']) # 모델 가중치 로드
    if optimizer and 'optimizer_state_dict' in checkpoint: # 옵티마이저와 저장된 상태가 있다면
        optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # 옵티마이저 상태 로드
    print(f"Checkpoint loaded from {filepath}") # 로드 완료 메시지
    return model, optimizer # 업데이트된 모델과 옵티마이저 반환

def calculate_metrics(targets, predictions_proba, threshold=0.5):
    """
    이진 분류 문제의 여러 평가 지표를 계산하는 함수입니다.
    Args:
        targets (numpy.array): 실제 정답 값 (0 또는 1).
        predictions_proba (numpy.array): 모델이 예측한 확률 값 (0~1 사이).
        threshold (float, optional): 확률 값을 0 또는 1로 변환하는 기준점. 기본값은 0.5.
    Returns:
        dict: 각 평가 지표의 이름과 값을 담은 딕셔너리.
    """
    # 확률 값을 기준으로 이진 예측(0 또는 1)을 만듭니다.
    predictions_binary = (predictions_proba >= threshold).astype(int)

    # 각 평가 지표 계산
    accuracy = accuracy_score(targets, predictions_binary) # 정확도
    # precision, recall, f1 계산 시, 한 클래스가 전혀 예측되지 않으면 경고가 뜨고 0이 반환될 수 있음.
    # zero_division=0 은 이 경우 0을 반환하도록 설정 (경고 대신)
    precision = precision_score(targets, predictions_binary, zero_division=0) # 정밀도
    recall = recall_score(targets, predictions_binary, zero_division=0)       # 재현율
    f1 = f1_score(targets, predictions_binary, zero_division=0)               # F1 점수
    
    try:
        # ROC AUC 점수는 실제 값과 '확률' 예측값을 사용합니다.
        roc_auc = roc_auc_score(targets, predictions_proba)
    except ValueError: # 만약 모든 타겟이 같은 값(예: 전부 0 또는 전부 1)이면 ROC AUC 계산 불가
        roc_auc = float('nan') # Not a Number (숫자 아님)으로 처리

    # 계산된 지표들을 딕셔너리 형태로 모아서 반환
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }
    return metrics