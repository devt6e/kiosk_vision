# dataset.py
# 슬라이딩 윈도우 기반 프레임 레벨 모음 분류 Dataset

import os
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch

# ──────────────────────────────────────────────
# 라벨 문자열 → 클래스 인덱스 매핑
# ──────────────────────────────────────────────
LABEL_TO_IDX = {
    "A"       : 0,   # ㅏ, ㅑ
    "O"       : 1,   # ㅗ, ㅛ
    "U"       : 2,   # ㅜ, ㅠ
    "EU"      : 3,   # ㅡ
    "I"       : 4,   # ㅣ, ㅖ, ㅒ
    "EO"      : 5,   # ㅓ, ㅕ
    "E"       : 6,   # ㅔ, ㅐ
    "SILENCE" : 7,   # 침묵
}

IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}
NUM_CLASSES  = len(LABEL_TO_IDX)   # 8


# ──────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────
def find_pairs(data_dir: str) -> list[dict]:
    """
    processed 디렉토리에서 frames/labels .npy 페어 탐색.
    'test' stem은 제외.

    반환:
        [{"stem": ..., "frames_path": ..., "labels_path": ...}, ...]
    """
    data_dir = Path(data_dir)
    frames_files = {
        f.stem.replace("_frames", ""): f
        for f in data_dir.glob("*_frames.npy")
    }
    labels_files = {
        f.stem.replace("_labels", ""): f
        for f in data_dir.glob("*_labels.npy")
    }

    pairs = []
    for stem in sorted(frames_files.keys()):
        if stem == "test":
            continue
        if stem not in labels_files:
            print(f"[WARN] labels 없음, 제외: {stem}")
            continue
        pairs.append({
            "stem"        : stem,
            "frames_path" : str(frames_files[stem]),
            "labels_path" : str(labels_files[stem]),
        })

    print(f"[INFO] 총 {len(pairs)}개 페어 발견")
    return pairs


def split_pairs(
    pairs: list[dict],
    val_ratio: float  = 0.1,
    test_ratio: float = 0.1,
    seed: int         = 42,
) -> tuple[list, list, list]:
    """
    페어 리스트를 train / val / test 로 분할.
    파일 단위로 분할 (프레임 누수 방지).
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(pairs)).tolist()

    n_test = max(1, int(len(pairs) * test_ratio))
    n_val  = max(1, int(len(pairs) * val_ratio))

    test_idx  = indices[:n_test]
    val_idx   = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    train = [pairs[i] for i in train_idx]
    val   = [pairs[i] for i in val_idx]
    test  = [pairs[i] for i in test_idx]

    print(f"[INFO] 분할 — train: {len(train)}, val: {len(val)}, test: {len(test)}")
    return train, val, test


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class VowelWindowDataset(Dataset):
    """
    슬라이딩 윈도우 기반 모음 분류 Dataset.

    윈도우 크기 : window_size = 7
    스트라이드  : stride = 2
    라벨        : 윈도우 중앙 프레임 (index = window_size // 2) 의 라벨

    입력 텐서 shape : (1, T, H, W) = (1, 7, 64, 64)
    라벨            : int (0~7)

    사용 예시:
        dataset = VowelWindowDataset(pairs, window_size=7, stride=2)
        frames, label = dataset[0]
        # frames: torch.Tensor (1, 7, 64, 64)
        # label : int
    """

    def __init__(
        self,
        pairs       : list[dict],
        window_size : int  = 7,
        stride      : int  = 2,
        augment     : bool = False,
    ):
        self.window_size = window_size
        self.stride      = stride
        self.augment     = augment
        self.half        = window_size // 2   # 중앙 인덱스 = 3

        # 전체 윈도우 인덱스 목록 구성
        # (파일 로딩 비용 절감 위해 파일별로 한 번만 로드)
        self.windows: list[tuple[np.ndarray, np.ndarray, int]] = []
        # (frames_array, labels_array, center_frame_idx)

        self._build_index(pairs)

    def _build_index(self, pairs: list[dict]):
        """
        각 파일을 로드하고 윈도우 인덱스를 미리 구성.
        메모리 절약을 위해 numpy array 참조만 저장.
        """
        total_windows = 0

        for pair in pairs:
            frames = np.load(pair["frames_path"])   # (T, 64, 64, 3)
            labels = np.load(pair["labels_path"], allow_pickle=True)  # (T,)

            T = len(frames)
            assert len(labels) == T, (
                f"프레임/라벨 길이 불일치: {pair['stem']} "
                f"frames={T}, labels={len(labels)}"
            )

            # 슬라이딩 윈도우 중앙 프레임 인덱스 목록
            # 중앙이 half 이상 & (T - half - 1) 이하여야 양쪽 프레임 확보 가능
            centers = range(self.half, T - self.half, self.stride)

            for center in centers:
                self.windows.append((frames, labels, center))

            total_windows += len(list(centers))

        print(f"[INFO] 총 윈도우 수: {total_windows}")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        frames, labels, center = self.windows[idx]

        # 윈도우 프레임 추출 (center - half ~ center + half + 1)
        start = center - self.half
        end   = center + self.half + 1
        window_frames = frames[start:end]   # (7, 64, 64, 3)

        # 라벨 (중앙 프레임)
        label_str = labels[center]
        label_idx = LABEL_TO_IDX.get(label_str, 7)   # 미등록 → SILENCE

        # 전처리
        window_frames = self._preprocess(window_frames)   # (1, 7, 64, 64)

        # 증강
        if self.augment:
            window_frames = self._augment(window_frames)

        return window_frames, label_idx

    def _preprocess(self, frames: np.ndarray) -> torch.Tensor:
        """
        (7, 64, 64, 3) uint8
        → 그레이스케일 변환
        → 정규화 (0~1 float32)
        → (1, 7, 64, 64) tensor  [C, T, H, W]
        """
        # RGB → 그레이스케일 (가중 평균)
        gray = (
            0.299 * frames[:, :, :, 0].astype(np.float32) +
            0.587 * frames[:, :, :, 1].astype(np.float32) +
            0.114 * frames[:, :, :, 2].astype(np.float32)
        )   # (7, 64, 64)

        # 정규화
        gray = gray / 255.0

        # 텐서 변환 + 채널 차원 추가
        tensor = torch.from_numpy(gray).unsqueeze(0)   # (1, 7, 64, 64)

        return tensor

    def _augment(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        학습용 간단 증강.
        - 수평 반전 (50%)
        - 밝기 조절 (±10%)
        """
        # 수평 반전
        if torch.rand(1).item() > 0.5:
            tensor = torch.flip(tensor, dims=[-1])   # W축 반전

        # 밝기 조절
        brightness = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
        tensor = torch.clamp(tensor * brightness, 0.0, 1.0)

        return tensor


# ──────────────────────────────────────────────
# DataLoader 생성 헬퍼
# ──────────────────────────────────────────────
def get_dataloaders(
    data_dir    : str,
    batch_size  : int  = 32,
    window_size : int  = 7,
    stride      : int  = 2,
    val_ratio   : float = 0.1,
    test_ratio  : float = 0.1,
    num_workers : int  = 4,
    seed        : int  = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    train / val / test DataLoader 반환.
    """
    pairs = find_pairs(data_dir)
    train_pairs, val_pairs, test_pairs = split_pairs(
        pairs, val_ratio, test_ratio, seed
    )

    train_dataset = VowelWindowDataset(
        train_pairs, window_size, stride, augment=True
    )
    val_dataset   = VowelWindowDataset(
        val_pairs, window_size, stride, augment=False
    )
    test_dataset  = VowelWindowDataset(
        test_pairs, window_size, stride, augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )

    return train_loader, val_loader, test_loader


# ──────────────────────────────────────────────
# 테스트
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import os
    DATA_DIR = os.path.expanduser("~/lip_reading/data/processed/")

    # 페어 탐색
    pairs = find_pairs(DATA_DIR)
    train_pairs, val_pairs, test_pairs = split_pairs(pairs)

    # 데이터셋 생성
    train_ds = VowelWindowDataset(train_pairs, window_size=7, stride=2, augment=True)
    val_ds   = VowelWindowDataset(val_pairs,   window_size=7, stride=2, augment=False)

    print(f"\ntrain 윈도우 수 : {len(train_ds)}")
    print(f"val   윈도우 수 : {len(val_ds)}")

    # 샘플 1개 확인
    frames, label = train_ds[0]
    print(f"\n샘플 확인")
    print(f"  frames shape : {frames.shape}")   # (1, 7, 64, 64)
    print(f"  frames dtype : {frames.dtype}")
    print(f"  frames range : {frames.min():.3f} ~ {frames.max():.3f}")
    print(f"  label        : {label} ({IDX_TO_LABEL[label]})")

    # DataLoader 배치 확인
    loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    batch_frames, batch_labels = next(iter(loader))
    print(f"\n배치 확인")
    print(f"  batch frames shape : {batch_frames.shape}")  # (32, 1, 7, 64, 64)
    print(f"  batch labels shape : {batch_labels.shape}")  # (32,)

    # 클래스 분포 확인
    print(f"\n클래스 분포 (train 전체)")
    all_labels = []
    for _, lbl in train_ds:
        all_labels.append(lbl)
    all_labels = np.array(all_labels)
    for idx, name in IDX_TO_LABEL.items():
        count = (all_labels == idx).sum()
        ratio = count / len(all_labels) * 100
        print(f"  [{idx}] {name:8s} : {count:6d} ({ratio:.1f}%)")
