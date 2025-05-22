# src/model.py
import torch
import torch.nn as nn # nn은 Neural Network(신경망)의 약자로, 신경망 구성 요소를 담고 있습니다.

# config.py를 직접 불러오지는 않습니다. 모델 생성 시 필요한 값들은
# train.py에서 config 값을 참조하여 모델 객체를 만들 때 인자로 전달받습니다.

class WideAndDeepModel(nn.Module): # PyTorch의 모든 모델은 nn.Module을 상속받아 만듭니다.
    def __init__(self, wide_input_dim, deep_input_dim, deep_hidden_dims, dropout_rate):
        """
        Wide & Deep 모델의 구조를 초기화(설계)하는 함수입니다.
        모델이 생성될 때 한 번 호출됩니다.

        Args:
            wide_input_dim (int): Wide 파트에 입력될 정보의 총 개수.
                                  (예: 수치 피처 수 + 장르 OHE 수 + 월 OHE 수 + 언어 OHE 수)
            deep_input_dim (int): Deep 파트에 입력될 정보의 개수 (BERT CLS 벡터의 차원, 예: 768).
            deep_hidden_dims (list): Deep 파트 내부의 중간 계산층(은닉층)들의 크기를 담은 리스트.
                                     예: [256, 128] -> 첫 번째 은닉층 256개, 두 번째 128개 뉴런.
            dropout_rate (float): Deep 파트에서 사용할 드롭아웃 비율 (과적합 방지용).
        """
        super(WideAndDeepModel, self).__init__() # 부모 클래스(nn.Module)의 초기화 함수를 먼저 호출

        # --- Wide 파트 설계 ---
        # Wide 파트는 입력받은 여러 정보를 하나의 값(Wide Logit)으로 요약하는 간단한 선형 계산층입니다.
        # nn.Linear(입력_개수, 출력_개수): 선형 변환 y = Wx + b 를 수행하는 층.
        self.wide_linear = nn.Linear(wide_input_dim, 1) # 입력은 wide_input_dim개, 출력은 1개

        # --- Deep 파트 설계 ---
        # Deep 파트는 여러 개의 계산층을 순서대로 쌓아서 만듭니다.
        deep_layers = [] # 각 층들을 담을 빈 리스트
        current_dim = deep_input_dim # 첫 번째 층의 입력 크기는 deep_input_dim (CLS 벡터 차원)

        # deep_hidden_dims 리스트에 정의된 크기대로 중간층(은닉층)들을 만듭니다.
        for hidden_dim in deep_hidden_dims:
            deep_layers.append(nn.Linear(current_dim, hidden_dim)) # 선형 계산층
            deep_layers.append(nn.ReLU()) # ReLU 활성화 함수 (비선형성 추가, 이전 설명 참고)
            if dropout_rate > 0: # 드롭아웃 비율이 0보다 크면
                deep_layers.append(nn.Dropout(dropout_rate)) # 드롭아웃 층 추가 (과적합 방지)
            current_dim = hidden_dim # 다음 층의 입력 크기는 현재 층의 출력 크기가 됩니다.
        
        # Deep 파트의 마지막 출력층: 여러 중간 계산을 거친 결과를 하나의 값(Deep Logit)으로 요약합니다.
        deep_layers.append(nn.Linear(current_dim, 1)) # 입력은 마지막 은닉층 크기, 출력은 1개

        # nn.Sequential: 리스트에 담긴 여러 층들을 순서대로 실행하는 하나의 묶음(컨테이너)으로 만듭니다.
        # *deep_layers 는 리스트의 내용물을 풀어서 전달하라는 의미입니다.
        self.deep_sequential = nn.Sequential(*deep_layers)

    def forward(self, batch):
        """
        실제 데이터가 모델을 통과하는 계산 과정을 정의하는 함수입니다.
        모델 객체를 함수처럼 호출할 때 (예: model(데이터)) 이 함수가 실행됩니다.

        Args:
            batch (dict): Dataset의 __getitem__에서 반환한 딕셔너리 형태의 데이터 묶음.
                          {'wide_input': 텐서, 'deep_input': 텐서} 를 포함합니다.
        Returns:
            torch.Tensor: 모델의 최종 예측 확률 값 (0~1 사이, (배치크기,) 모양의 텐서).
        """
        # 딕셔너리에서 Wide 파트 입력과 Deep 파트 입력을 꺼냅니다.
        wide_input = batch['wide_input']
        deep_input = batch['deep_input']

        # 1. Wide 파트 계산: wide_input을 wide_linear 층에 통과시켜 wide_logit을 얻습니다.
        wide_logit = self.wide_linear(wide_input) # 모양: (배치크기, 1)

        # 2. Deep 파트 계산: deep_input을 deep_sequential 층 묶음에 통과시켜 deep_logit을 얻습니다.
        deep_logit = self.deep_sequential(deep_input) # 모양: (배치크기, 1)

        # 3. Logits 결합: Wide Logit과 Deep Logit을 더합니다. (가장 간단한 결합 방식)
        combined_logit = wide_logit + deep_logit # 모양: (배치크기, 1)

        # 4. 최종 예측 확률 계산: 결합된 logit에 시그모이드(Sigmoid) 함수를 적용합니다.
        # 시그모이드 함수는 어떤 숫자든 0과 1 사이의 값으로 바꿔줍니다 (확률처럼).
        output = torch.sigmoid(combined_logit) # 모양: (배치크기, 1)

        # .squeeze(-1): 텐서의 마지막 차원 중 크기가 1인 것을 제거합니다.
        # (배치크기, 1) -> (배치크기,) 모양으로 변경.
        # 이는 나중에 손실 함수(BCELoss)가 기대하는 입력 형태와 맞추기 위함입니다.
        return output.squeeze(-1)

if __name__ == '__main__':
    # 이 파일을 직접 실행할 때 (python src/model.py) 아래 코드가 실행됩니다. (테스트용)
    print("Running model.py as a script...")
    
    # config.py에서 모델 설정값을 가져오거나, 테스트용 임의의 값을 사용합니다.
    # 실제로는 train.py에서 Dataset을 통해 얻은 wide_input_dim을 사용해야 합니다.
    # 여기서는 테스트를 위해 임의의 값을 가정합니다.
    try:
        from . import config # config.py 임포트 시도
        _deep_input_dim = config.DEEP_INPUT_DIM
        _deep_hidden_dims = config.DEEP_HIDDEN_DIMS
        _dropout_rate = config.DROPOUT_RATE
        # WIDE_INPUT_DIM은 Dataset에서 결정되므로, 여기서는 임의의 값을 사용합니다.
        # 실제 학습 시에는 이 값이 정확해야 합니다.
        _test_wide_dim = 50 # 예시: 수치 7개 + 장르 20개 + 월 12개 + 언어 10개 = 49개 (대략적인 값), 실제로 122개개
        print(f"Using DEEP_INPUT_DIM: {_deep_input_dim}, DEEP_HIDDEN_DIMS: {_deep_hidden_dims}, DROPOUT_RATE: {_dropout_rate}")
        print(f"For model structure test, using an arbitrary WIDE_INPUT_DIM: {_test_wide_dim}")
    except ImportError: # 만약 config.py를 불러올 수 없다면 (예: 단독 실행 시 경로 문제)
        print("config.py not found or import error. Using default test parameters for model structure.")
        _test_wide_dim = 50
        _deep_input_dim = 768
        _deep_hidden_dims = [256, 128]
        _dropout_rate = 0.2

    # 모델 객체를 만듭니다.
    model_instance = WideAndDeepModel(
        wide_input_dim=_test_wide_dim,
        deep_input_dim=_deep_input_dim,
        deep_hidden_dims=_deep_hidden_dims,
        dropout_rate=_dropout_rate
    )
    print("\n--- WideAndDeepModel Architecture ---")
    print(model_instance) # 모델의 전체 구조를 출력해봅니다.

    # 가상의 입력 데이터(더미 데이터)로 forward 함수가 잘 동작하는지 테스트해봅니다.
    batch_size_test = 4 # 4개의 샘플이 한 묶음(배치)이라고 가정
    # wide_input은 (배치크기, wide_input_dim) 모양의 랜덤 텐서
    dummy_wide_input = torch.randn(batch_size_test, _test_wide_dim)
    # deep_input은 (배치크기, deep_input_dim) 모양의 랜덤 텐서
    dummy_deep_input = torch.randn(batch_size_test, _deep_input_dim)
    # 모델 입력은 딕셔너리 형태
    dummy_batch = {'wide_input': dummy_wide_input, 'deep_input': dummy_deep_input}

    try:
        # 모델에 더미 배치를 넣어 예측값을 얻어봅니다 (forward 함수 호출)
        predictions = model_instance(dummy_batch)
        print("\n--- Dummy Forward Pass Test ---")
        print(f"Input batch size: {batch_size_test}")
        print(f"Output predictions shape: {predictions.shape}") # (4,) 모양이어야 함
        print(f"Sample predictions (probabilities 0-1): {predictions}")
    except Exception as e:
        print(f"Error during dummy forward pass: {e}")
        import traceback
        traceback.print_exc()