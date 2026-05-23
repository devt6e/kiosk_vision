# kiosk_vision
# 한국어 모음 립리딩 시스템

소음 환경에서 Google STT의 보조 수단으로 입술 모양을 분석해 한국어 모음을 분류하는 딥러닝 모델.

---

## 모음 분류 클래스 (8클래스)

| 인덱스 | 모음 | 로마자 |
|--------|------|--------|
| 0 | ㅏ, ㅑ | A |
| 1 | ㅗ, ㅛ | O |
| 2 | ㅜ, ㅠ | U |
| 3 | ㅡ | EU |
| 4 | ㅣ, ㅖ, ㅒ | I |
| 5 | ㅓ, ㅕ | EO |
| 6 | ㅔ, ㅐ | E |
| 7 | 침묵 | SILENCE |

---

## 시스템 구조

```
[Windows]
카메라 (640×480)
    ↓
MediaPipe FaceLandmarker → 입술 크롭 (64×64)
    ↓
TCP 소켓 전송 (15fps)
    ↓
[WSL / Linux]
UtteranceDetector (모션 기반 발화 감지)
    ↓
VowelInferencer
  VisualFrontend (Conv3d × 3)
  → VisualTransformerEncoder (d_model=512, nhead=8, layers=4)
  → ClassificationHead (8클래스)
    ↓
모음 시퀀스 출력 ['A', 'EO', 'I', ...]
```

---

## 파일 구조

```
lip_reading/
├── scripts/
│   ├── config.py          # 하이퍼파라미터, 클래스 정의
│   ├── vocab.py           # VowelVocab (텍스트 → 모음 변환)
│   ├── dataset.py         # 슬라이딩 윈도우 Dataset (window=7, stride=2)
│   ├── model.py           # VowelClassifier (Conv3d + Transformer)
│   ├── train.py           # 학습 루프 (클래스 가중치 + Cosine 스케줄러)
│   ├── inference.py       # 실시간 추론 파이프라인
│   └── camera_server.py   # Windows 카메라 서버 (MediaPipe 시각화)
├── checkpoints/
│   ├── best.pt            # 최고 val_loss 체크포인트
│   └── latest.pt          # 최근 체크포인트
├── models/
│   └── face_landmarker.task  # MediaPipe FaceLandmarker 모델
└── data/
    └── processed/         # 전처리된 .npy 파일 (frames + labels)
```

---

## 환경 설정

### 요구사항

```
Python 3.12+
CUDA 지원 GPU (권장)
```

### 패키지 설치

```bash
pip install torch torchvision
pip install mediapipe
pip install opencv-python
pip install numpy
```

### MediaPipe 모델 다운로드

```bash
mkdir -p ~/lip_reading/models
cd ~/lip_reading/models
wget https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

---

## 데이터 준비

### 데이터 형식

```
data/processed/
├── {파일명}_frames.npy   # shape: (T, 64, 64, 3), dtype: uint8
└── {파일명}_labels.npy   # shape: (T,), dtype: str
                          # 값: 'A', 'O', 'U', 'EU', 'I', 'EO', 'E', 'SILENCE'
```

### 데이터 확인

```python
import numpy as np

frames = np.load("data/processed/sample_frames.npy")
labels = np.load("data/processed/sample_labels.npy", allow_pickle=True)

print(frames.shape)   # (T, 64, 64, 3)
print(np.unique(labels))  # ['A' 'E' 'EO' 'EU' 'I' 'O' 'SILENCE' 'U']
```

---

## 학습

### 기본 실행

```bash
cd ~/lip_reading/scripts
python train.py
```

### 주요 설정 (train.py CFG)

```python
CFG = {
    "window_size" : 7,      # 슬라이딩 윈도우 크기
    "stride"      : 5,      # 슬라이딩 윈도우 스트라이드
    "batch_size"  : 64,
    "learning_rate": 1e-4,
    "epochs"      : 50,
    "warmup_epochs": 5,
    "use_class_weight"   : True,   # 클래스 불균형 처리
    "use_weighted_sampler": False,
}
```

### 학습 결과 확인

```
체크포인트: ~/lip_reading/checkpoints/best.pt
```

---

## 추론

### WSL + Windows 소켓 방식 (개발/테스트용)

**① Windows PowerShell — 카메라 서버 실행**

```powershell
cd C:\...\lip_reading\scripts
python camera_server.py
```

MediaPipe가 없으면
```powershell
python camera_server.py --no-mediapipe
```

**② WSL — 추론 실행**

```bash
cd ~/lip_reading/scripts
python inference.py --socket --no-display
```

디버그 모드 (모션값 실시간 확인)
```bash
python inference.py --socket --no-display --debug
```

### 로컬 카메라 방식 (Linux 단독)

```bash
python inference.py
```

### .npy 파일로 테스트

```bash
python inference.py --test-file ~/lip_reading/data/processed/sample_frames.npy
```

---

## 추론 파라미터 조정

`inference.py` 상단 `INF_CFG`에서 조정 가능.

```python
INF_CFG = {
    "motion_threshold"  : 10.0,  # 높일수록 발화 감지 기준 엄격
    "speech_trigger_thr": 3,     # 발화 시작까지 연속 감지 필요 프레임 수
    "silence_duration"  : 0.8,   # 발화 종료 판단 침묵 시간 (초)
    "min_speech_frames" : 10,    # 유효 발화 최소 프레임 수
    "max_speech_frames" : 150,   # 최대 발화 프레임 (강제 종료)
}
```

---

## 모델 구조

```
입력: (B, 1, 7, 64, 64)

VisualFrontend (Conv3d × 3)
  Conv3d(1→32,   stride=(1,2,2))  → (B, 32,  7, 32, 32)
  Conv3d(32→64,  stride=(1,2,2))  → (B, 64,  7, 16, 16)
  Conv3d(64→128, stride=(1,2,2))  → (B, 128, 7,  8,  8)

SpatialPool + Linear
  AdaptiveAvgPool3d → (B, 128, 7, 1, 1)
  Linear(128→512)   → (B, 7, 512)

VisualTransformerEncoder
  d_model=512, nhead=8, layers=4
  시간 평균 풀링     → (B, 512)

ClassificationHead
  Linear(512→256) → ReLU → Linear(256→8)

출력: (B, 8)  ← 8클래스 logits
```

파라미터 수: 약 13M

---

## 성능

| 조건 | val_acc |
|------|---------|
| 90개 파일, epoch 28 | 40.5% |
| 파인튜닝 예정 (500개 추가) | - |

> 립리딩 단독 정확도보다 **STT + 립리딩 융합 정확도**가 실제 지표.

---

## 향후 계획

```
① 파인튜닝 데이터 수집 (2인 × 250개 = 500개)
② 추가 학습으로 정확도 개선
③ Spring Boot REST API 연동 (kiosk_bridge)
④ STT 결과 + 모음 시퀀스 융합 매칭
```
