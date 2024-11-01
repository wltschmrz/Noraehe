# utils/path_config.py
from pathlib import Path

# project root dir
ROOT_DIR = Path(__file__).resolve().parent.parent

# dir paths
DATA_DIR = ROOT_DIR / 'data'
FAISS_DIR = DATA_DIR / 'faiss'
TEMP_DIR = DATA_DIR / 'temporary_folder'  # 임시 폴더 이름
SRC_DIR = ROOT_DIR / 'src'
PITCH_DETECTING_DIR = SRC_DIR / 'pitch_detecting'
TIMBRE_ENCODING_DIR = SRC_DIR / 'timbre_encoding'
MODELS_DIR = TIMBRE_ENCODING_DIR / 'models'
PRETRAINED_TIMBRE_DIR = TIMBRE_ENCODING_DIR / 'pretrained_timbre_enc'
UTILS_DIR = ROOT_DIR / 'utils'

# file paths
INDEX_TITLE_FILE = FAISS_DIR / 'index_title.bin'
INDEX_TIMBRE_FILE = FAISS_DIR / 'index_timbre.bin'
SETS_PATH_FILE = FAISS_DIR / 'sets_path.pkl'

# pitch_detecting module files
AUDIO_PROCESSOR_FILE = PITCH_DETECTING_DIR / 'audio_processor.py'
EMBEDDING_FILE = PITCH_DETECTING_DIR / 'embedding.py'
FAISS_INDEX_FILE = PITCH_DETECTING_DIR / 'faiss_index.py'
VOCAL_RANGE_FILE = PITCH_DETECTING_DIR / 'vocal_range.py'

# timbre_encoding module files
LSTM_MODEL_FILE = MODELS_DIR / 'lstm.py'
TIMBRE_ENCODER_FILE = TIMBRE_ENCODING_DIR / 'timbre_encoder.py'
PRETRAINED_MODEL_FILE = PRETRAINED_TIMBRE_DIR / 'best_model.pth.tar'
PRETRAINED_CONFIG_FILE = PRETRAINED_TIMBRE_DIR / 'config.json'

# utils module files
AUDIO_UTIL_FILE = UTILS_DIR / 'audio.py'
CONFIG_UTIL_FILE = UTILS_DIR / 'config.py'
COGPIT_UTIL_FILE = UTILS_DIR / 'cogpit.py'
IO_UTIL_FILE = UTILS_DIR / 'io.py'
READ_JSON_UTIL_FILE = UTILS_DIR / 'read_json.py'
SHARED_CONFIGS_FILE = UTILS_DIR / 'shared_configs.py'

# additional project files
MAIN_FILE = ROOT_DIR / 'main.py'
REQUIREMENTS_FILE = ROOT_DIR / 'requirements.txt'
NOTEBOOK_FILE = ROOT_DIR / 'run.ipynb'
STACK_DATA_FILE = ROOT_DIR / 'stack_data.py'

# 현재 프로젝트 구조
'''
/Noraehe
├── data/
│   ├── faiss/
│   │   ├── index_title.bin
│   │   ├── index_timbre.bin
│   │   └── sets_path.pkl
│   └── temporary_folder/ (폴더명 기억안남)
│
├── src/
│   ├── pitch_detecting/
│   │   ├── audio_processor.py
│   │   ├── embedding.py
│   │   ├── faiss_index.py
│   │   └── vocal_range.py
│   │
│   └── timbre_encoding/
│       ├── models/
│       │   └── lstm.py
│       ├── pretrained_timbre_enc/
│       │   ├── best_model.pth.tar
│       │   └── config.json
│       └── timbre_encoder.py
│
├── utils/
│   ├── audio.py
│   ├── config.py
│   ├── cogpit.py
│   ├── io.py
│   ├── path_config.py       # 현 위치
│   ├── read_json.py
│   └── shared_configs.py
│
├── main.py
├── requirements.txt
├── run.ipynb
└── stack_data.py
'''


# Project Structure - (이렇게 바꿀 예정)
'''
Noraehe/
│
├── app/                # 웹 애플리케이션, API 서버 관련 코드
│   └── main.py         # 프로젝트의 메인 실행 파일
│
├── src/                # 핵심 비즈니스 로직 코드
│   ├── models/         # 모델 정의 코드 (딥러닝 모델 등)
│   ├── processing/     # 데이터 처리 및 전처리 코드
│   ├── services/       # 애플리케이션 서비스 로직
│   ├── inference/      # 추론 및 예측 코드
│   └── utils/          # 공통 유틸리티 함수
│
├── data/               # 데이터 저장소 (원본, 가공, 외부 데이터)
│   ├── raw/            # 원본 데이터
│   ├── processed/      # 전처리된 데이터
│   └── external/       # 외부에서 가져온 데이터셋
│
├── config/             # 설정 파일 (파라미터, 경로, 환경 변수)
│   ├── model_config.yaml  # 모델 관련 설정 파일
│   └── app_config.yaml    # 애플리케이션 설정 파일
│
├── docs/               # 문서화 파일 (예: API 문서, 개발 가이드)
│   ├── README.md       # 프로젝트 설명 파일
│   └── CHANGELOG.md    # 변경 내역
│
├── requirements.txt    # 파이썬 패키지 의존성 목록
└── setup.py            # 패키지화 설정 파일 (라이브러리 형태일 경우)
'''