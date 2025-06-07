# src/dataset.py
import torch
from torch.utils.data import Dataset # PyTorch의 Dataset 클래스를 가져옵니다.
import pandas as pd
import numpy as np
import os # 파일 존재 여부 확인 등
from . import config # 우리 프로젝트의 설정값을 담고 있는 config.py를 불러옵니다.
import joblib # 저장된 MLB, OHE 객체를 불러오기 위해 필요

class MovieSuccessDataset(Dataset):
    def __init__(self, data_path, mode='train'): # 생성자 함수, 객체 만들 때 처음 호출됨
        """
        Dataset 객체를 초기화합니다. 전처리된 데이터를 불러오고,
        모델에 입력될 각 부분(Wide, Deep, Target)을 준비합니다.

        Args:
            data_path (str): 전처리된 데이터 파일(.pkl)의 경로.
            mode (str): 현재 모드 ('train', 'evaluate', 'predict').
                        (여기서는 주로 wide_feature_cols 결정에 간접적으로 영향)
        """
        if not os.path.exists(data_path): # 데이터 파일이 없으면 오류 발생
            raise FileNotFoundError(f"Processed data file not found at {data_path}. Please run preprocessing first.")
        
        self.dataframe = pd.read_pickle(data_path) # Pickle 파일에서 전처리된 DataFrame을 불러옵니다.
        self.mode = mode

        # --- Wide 파트에 사용될 피처 컬럼 이름들을 결정합니다 ---
        # 1. 수치형 피처 (config.py 에서 가져옴)
        numerical_cols = config.NUMERICAL_FEATURES

        # 2. 장르 피처 (저장된 MLB 객체에서 클래스 이름을 가져와 생성)
        genre_cols = []
        if os.path.exists(config.MLB_PATH):
            mlb = joblib.load(config.MLB_PATH)
            genre_cols = [f"{config.GENRE_MLB_PREFIX}{cls}" for cls in mlb.classes_]
        elif self.mode != 'train': # 학습 모드가 아닌데 MLB 파일이 없으면, DataFrame에서 유추 시도
            genre_cols = [col for col in self.dataframe.columns if col.startswith(config.GENRE_MLB_PREFIX)]
            if not genre_cols: print(f"Warning: MLB file not found and no genre columns found in DataFrame for mode '{self.mode}'.")
        else: # 학습 모드인데 MLB 파일이 없으면 오류 (preprocess.py에서 생성되어야 함)
            raise FileNotFoundError(f"MLB file not found at {config.MLB_PATH}. Required for training mode.")

        # 3. 월(Month) 원-핫 인코딩 피처 (저장된 OHE 객체에서 가져옴)
        month_ohe_cols = []
        if os.path.exists(config.MONTH_OHE_PATH):
            month_ohe = joblib.load(config.MONTH_OHE_PATH)
            # OneHotEncoder의 categories_는 리스트의 리스트 형태이므로 첫 번째 요소 사용
            month_ohe_cols = [f"{config.MONTH_OHE_PREFIX}{int(cat)}" for cat in month_ohe.categories_[0]]
        elif self.mode != 'train':
            month_ohe_cols = [col for col in self.dataframe.columns if col.startswith(config.MONTH_OHE_PREFIX)]
            if not month_ohe_cols: print(f"Warning: Month OHE file not found and no month columns found for mode '{self.mode}'.")
        else:
            raise FileNotFoundError(f"Month OHE file not found at {config.MONTH_OHE_PATH}. Required for training mode.")

        # 4. 언어(Language) 원-핫 인코딩 피처 (저장된 OHE 객체에서 가져옴)
        lang_ohe_cols = []
        if os.path.exists(config.LANGUAGE_OHE_PATH):
            lang_ohe = joblib.load(config.LANGUAGE_OHE_PATH)
            lang_ohe_cols = [f"{config.LANGUAGE_OHE_PREFIX}{cat}" for cat in lang_ohe.categories_[0]]
        elif self.mode != 'train':
            lang_ohe_cols = [col for col in self.dataframe.columns if col.startswith(config.LANGUAGE_OHE_PREFIX)]
            if not lang_ohe_cols: print(f"Warning: Language OHE file not found and no language columns found for mode '{self.mode}'.")
        else:
            raise FileNotFoundError(f"Language OHE file not found at {config.LANGUAGE_OHE_PATH}. Required for training mode.")
        
        prodco_cols = []
        # --- 제작사 피처 컬럼명 로드 ---
        prodco_mlb_path = config.PRODCO_MLB_PATH # config에서 경로 가져오기
        if os.path.exists(prodco_mlb_path):
            prodco_mlb = joblib.load(prodco_mlb_path)
            prodco_cols = [f"{config.PRODUCTION_COMPANY_MLB_PREFIX}{cls}" for cls in prodco_mlb.classes_]
        elif self.mode != 'train':
             prodco_cols = [col for col in self.dataframe.columns if col.startswith(config.PRODUCTION_COMPANY_MLB_PREFIX)]
             if not prodco_cols: print(f"Warning: Production Company MLB file not found and no prodco columns found for mode '{self.mode}'.")
        else:
             raise FileNotFoundError(f"Production Company MLB file not found at {prodco_mlb_path}. Required for training mode.")

        # 모든 Wide 파트 피처 컬럼들을 합칩니다.
        self.wide_feature_cols = numerical_cols + genre_cols + month_ohe_cols + lang_ohe_cols + prodco_cols
        # print(f"Dataset - Determined wide_feature_cols: {self.wide_feature_cols}") # 디버깅용

        # --- 모델 입력용 데이터 미리 추출 및 타입 변환 (효율성 증대) ---
        # Wide 파트 입력 데이터: 위에서 결정된 wide_feature_cols에 해당하는 열들을 가져와 NumPy 배열로 만듦
        # .astype(np.float32)는 숫자들을 32비트 실수형으로 통일 (PyTorch가 선호)
        self.wide_features_data = self.dataframe[self.wide_feature_cols].values.astype(np.float32)

        # Deep 파트 입력 데이터: CLS 벡터가 담긴 열(config.CLS_VECTOR_COL)을 가져옴
        # CLS 벡터는 이미 NumPy 배열이므로, 이 배열들을 쌓아서(stack) 하나의 큰 배열로 만듦
        self.deep_features_data = np.stack(self.dataframe[config.CLS_VECTOR_COL].values).astype(np.float32)

        # 타겟(정답) 데이터: 'success' 열(config.TARGET_COL)을 가져옴
        self.targets_data = self.dataframe[config.TARGET_COL].values.astype(np.float32)

        # 실제 Wide 파트 입력 차원 (나중에 모델 만들 때 사용)
        self.actual_wide_input_dim = self.wide_features_data.shape[1]

    def __len__(self):
        """이 Dataset에 들어있는 총 데이터 샘플의 개수를 반환합니다."""
        return len(self.dataframe) # DataFrame의 행(row) 개수와 동일

    def __getitem__(self, idx):
        """
        주어진 번호(idx)에 해당하는 하나의 데이터 샘플을 꺼내주는 함수입니다.
        마치 리스트에서 list[idx] 와 같이 특정 위치의 항목을 가져오는 것과 같습니다.
        """
        # 미리 준비해둔 NumPy 배열에서 idx 번째 데이터를 가져옵니다.
        wide_input = torch.tensor(self.wide_features_data[idx], dtype=torch.float)
        deep_input = torch.tensor(self.deep_features_data[idx], dtype=torch.float)
        target = torch.tensor(self.targets_data[idx], dtype=torch.float)

        # 가져온 데이터들을 딕셔너리(이름표가 붙은 꾸러미) 형태로 묶어서 반환합니다.
        # 나중에 모델 학습 시 이 이름표('wide_input', 'deep_input', 'target')를 사용해 데이터를 꺼냅니다.
        return {
            'wide_input': wide_input,
            'deep_input': deep_input,
            'target': target
        }

if __name__ == '__main__':
    # 이 파일을 직접 실행할 때 (python src/dataset.py) 아래 코드가 실행됩니다. (테스트용)
    print("Running dataset.py as a script...")
    # 전처리된 데이터 파일이 config.PROCESSED_DATA_PATH 경로에 있어야 합니다.
    # (없다면 먼저 python -m src.preprocess 를 실행해야 합니다.)
    if not os.path.exists(config.PROCESSED_DATA_PATH):
        print(f"Error: Processed data file not found at {config.PROCESSED_DATA_PATH}")
        print("Please run 'python -m src.preprocess' first to generate the processed data.")
    elif not os.path.exists(config.MLB_PATH) or \
         not os.path.exists(config.MONTH_OHE_PATH) or \
         not os.path.exists(config.LANGUAGE_OHE_PATH):
        print(f"Error: One or more preprocessor files (MLB, Month OHE, Language OHE) not found in {config.MODEL_DIR}.")
        print("Please ensure preprocessing was completed successfully in 'train' mode.")
    else:
        try:
            dataset = MovieSuccessDataset(data_path=config.PROCESSED_DATA_PATH, mode='evaluate') # 테스트 시 evaluate 모드
            print(f"Dataset loaded successfully. Number of samples: {len(dataset)}")
            print(f"Determined wide_input_dim from dataset: {dataset.actual_wide_input_dim}")

            # 첫 번째 샘플을 가져와서 내용 확인
            if len(dataset) > 0:
                sample = dataset[0]
                print("\n--- First Sample Retrieved from Dataset ---")
                print(f"Wide input shape: {sample['wide_input'].shape}, dtype: {sample['wide_input'].dtype}")
                print(f"Deep input shape: {sample['deep_input'].shape}, dtype: {sample['deep_input'].dtype}")
                print(f"Target: {sample['target']}, dtype: {sample['target'].dtype}")
            else:
                print("Dataset is empty.")
        except Exception as e:
            print(f"Error during dataset test: {e}")
            import traceback
            traceback.print_exc()