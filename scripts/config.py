# config.py
# 모음 분류 립리딩 시스템 — 전체 설정

from dataclasses import dataclass, field
from pathlib import Path

# ──────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────
ROOT_DIR       = Path(__file__).resolve().parent.parent
DATA_DIR       = ROOT_DIR / "data"
RAW_DIR        = DATA_DIR / "raw"
CACHE_DIR      = DATA_DIR / "cache"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
MODEL_DIR      = ROOT_DIR / "models"

# ──────────────────────────────────────────────
# 모음 클래스 정의 (8클래스 + blank)
# ──────────────────────────────────────────────
# CTC blank index = 0 (관례)
# 실제 모음 클래스 index = 1~8

VOWEL_CLASSES = {
    1: ["ㅏ", "ㅑ"],
    2: ["ㅗ", "ㅛ"],
    3: ["ㅜ", "ㅠ"],
    4: ["ㅡ"],
    5: ["ㅣ", "ㅖ", "ㅒ"],
    6: ["ㅓ", "ㅕ"],
    7: ["ㅔ", "ㅐ"],
    8: ["침묵"],   # silence
}

VOWEL_CLASS_NAMES = {
    0: "blank",
    1: "ㅏ/ㅑ",
    2: "ㅗ/ㅛ",
    3: "ㅜ/ㅠ",
    4: "ㅡ",
    5: "ㅣ/ㅖ/ㅒ",
    6: "ㅓ/ㅕ",
    7: "ㅔ/ㅐ",
    8: "침묵",
}

VOCAB_SIZE = 9  # blank(0) + 8클래스


# ──────────────────────────────────────────────
# 데이터 설정
# ──────────────────────────────────────────────
@dataclass
class DataConfig:
    # AI Hub 데이터 각도 필터 (A=정면, I=정면상단)
    valid_angles: tuple = ("A", "I")

    # 입술 크롭 크기
    lip_height: int = 96
    lip_width:  int = 96

    # 영상 프레임 설정
    target_fps:  int = 15
    max_frames:  int = 48
    min_frames:  int = 5

    # 데이터 분할
    val_ratio:   float = 0.1
    test_ratio:  float = 0.1
    random_seed: int   = 42

    # AI Hub JSON 구조 키
    sentence_info_key:     str = "Sentence_info"
    bounding_box_info_key: str = "Bounding_box_info"
    lip_bbox_key:          str = "Lip_bounding_box"
    bbox_coord_key:        str = "xtl_ytl_xbr_ybr"


# ──────────────────────────────────────────────
# 모델 설정
# ──────────────────────────────────────────────
@dataclass
class ModelConfig:
    # VisualFrontend (Conv3d)
    in_channels:  int = 1          # 그레이스케일
    frontend_out: int = 512        # frontend 출력 채널

    # VisualTransformerEncoder
    d_model:            int = 512
    nhead:              int = 8
    num_encoder_layers: int = 6
    dim_feedforward:    int = 2048
    dropout:            float = 0.1

    # CTC Head
    vocab_size: int = VOCAB_SIZE   # 9 (blank + 8클래스)

    # Temporal Pooling
    pool_type: str = "none"        # CTC는 시퀀스 유지, pooling 없음


# ──────────────────────────────────────────────
# 학습 설정
# ──────────────────────────────────────────────
@dataclass
class TrainConfig:
    batch_size:    int   = 4
    learning_rate: float = 1e-4
    epochs:        int   = 50
    warmup_epochs: int   = 5

    # CTC Loss
    ctc_zero_infinity: bool = True

    # 체크포인트
    save_every:    int  = 5         # N 에폭마다 저장
    early_stop:    int  = 10        # val_loss 개선 없으면 종료

    # 기기
    device: str = "cuda"            # "cuda" or "cpu"

    # DataLoader
    num_workers: int = 4
    pin_memory:  bool = True


# ──────────────────────────────────────────────
# 추론 설정
# ──────────────────────────────────────────────
@dataclass
class InferenceConfig:
    # 발화 감지 (B방식 — 발화 단위)
    silence_threshold:   float = 0.6    # 침묵 클래스 확률 임계값
    min_silence_frames:  int   = 8      # 침묵으로 판단할 최소 프레임 수
    min_speech_frames:   int   = 5      # 유효 발화로 판단할 최소 프레임 수

    # 카메라
    camera_index: int = 0
    frame_buffer: int = 48              # 수집할 최대 프레임 수

    # 소켓 (WSL ↔ Windows)
    socket_port:    int = 9999
    socket_timeout: int = 10


# ──────────────────────────────────────────────
# 전역 인스턴스 (import해서 바로 사용)
# ──────────────────────────────────────────────
DATA_CFG      = DataConfig()
MODEL_CFG     = ModelConfig()
TRAIN_CFG     = TrainConfig()
INFERENCE_CFG = InferenceConfig()


# ──────────────────────────────────────────────
# 디렉토리 자동 생성
# ──────────────────────────────────────────────
def init_dirs():
    for d in [DATA_DIR, RAW_DIR, CACHE_DIR, CHECKPOINT_DIR, MODEL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    init_dirs()
    print("=== 설정 확인 ===")
    print(f"VOCAB_SIZE : {VOCAB_SIZE}")
    print(f"클래스 목록:")
    for idx, name in VOWEL_CLASS_NAMES.items():
        print(f"  [{idx}] {name}")
    print(f"\nDataConfig  : {DATA_CFG}")
    print(f"ModelConfig : {MODEL_CFG}")
    print(f"TrainConfig : {TRAIN_CFG}")
    print(f"InferenceConfig : {INFERENCE_CFG}")
