# train.py
# 모음 분류 모델 학습 루프

import os
import time
import numpy as np
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import VowelWindowDataset, find_pairs, split_pairs, LABEL_TO_IDX, IDX_TO_LABEL
from model import VowelClassifier

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
DATA_DIR       = os.path.expanduser("~/lip_reading/data/processed/")
CHECKPOINT_DIR = os.path.expanduser("~/lip_reading/checkpoints/")

CFG = {
    # 데이터
    "window_size" : 7,
    "stride"      : 5,
    "val_ratio"   : 0.1,
    "test_ratio"  : 0.1,
    "seed"        : 42,

    # 모델
    "d_model"            : 512,
    "nhead"              : 8,
    "num_encoder_layers" : 4,
    "dim_feedforward"    : 2048,
    "dropout"            : 0.1,
    "num_classes"        : 8,

    # 학습
    "batch_size"    : 64,
    "learning_rate" : 1e-4,
    "weight_decay"  : 1e-4,
    "epochs"        : 50,
    "warmup_epochs" : 5,
    "early_stop"    : 10,     # val_loss 개선 없으면 종료
    "num_workers"   : 4,

    # 클래스 불균형
    "use_class_weight"   : True,   # Loss 가중치
    "use_weighted_sampler": False,  # 배치 샘플링 균등화
}


# ──────────────────────────────────────────────
# 클래스 가중치 계산
# ──────────────────────────────────────────────
def compute_class_weights(dataset: VowelWindowDataset, device: torch.device) -> torch.Tensor:
    """
    학습 데이터 클래스 분포 → 역빈도 가중치 계산.
    적은 클래스일수록 높은 가중치.
    """
    print("[INFO] 클래스 가중치 계산 중...")
    label_counts = Counter()

    for _, label in dataset:
        label_counts[label] += 1

    total = sum(label_counts.values())
    num_classes = CFG["num_classes"]

    weights = []
    for i in range(num_classes):
        count = label_counts.get(i, 1)
        w = total / (num_classes * count)
        weights.append(w)
        print(f"  [{i}] {IDX_TO_LABEL[i]:8s}: {count:7,}개  weight={w:.3f}")

    return torch.tensor(weights, dtype=torch.float32).to(device)


def compute_sample_weights(dataset: VowelWindowDataset) -> list[float]:
    """
    WeightedRandomSampler용 샘플별 가중치 계산.
    """
    print("[INFO] 샘플 가중치 계산 중...")
    label_counts = Counter()
    labels = []

    for _, label in dataset:
        label_counts[label] += 1
        labels.append(label)

    weight_per_class = {
        cls: 1.0 / count for cls, count in label_counts.items()
    }
    sample_weights = [weight_per_class[lbl] for lbl in labels]
    return sample_weights


# ──────────────────────────────────────────────
# Warmup 스케줄러
# ──────────────────────────────────────────────
class WarmupCosineScheduler:
    """
    Warmup 이후 Cosine Annealing 적용.
    warmup_epochs 동안 lr: 0 → base_lr 선형 증가
    이후 CosineAnnealingLR로 감소
    """

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, base_lr: float):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr       = base_lr
        self.cosine        = CosineAnnealingLR(
            optimizer,
            T_max = total_epochs - warmup_epochs,
            eta_min = base_lr * 0.01,
        )
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
        else:
            self.cosine.step()

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


# ──────────────────────────────────────────────
# 학습 / 검증 1 에폭
# ──────────────────────────────────────────────
def run_epoch(
    model     : nn.Module,
    loader    : DataLoader,
    criterion : nn.Module,
    optimizer : torch.optim.Optimizer | None,
    device    : torch.device,
    is_train  : bool,
) -> tuple[float, float]:
    """
    1 에폭 학습 또는 검증.
    반환: (loss, accuracy)
    """
    model.train() if is_train else model.eval()

    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for frames, labels in loader:
            frames = frames.to(device)
            labels = labels.to(device)

            logits = model(frames)               # (B, 8)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # 통계
            preds          = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            total_loss    += loss.item() * labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples * 100

    return avg_loss, accuracy


# ──────────────────────────────────────────────
# 체크포인트
# ──────────────────────────────────────────────
def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model: nn.Module, optimizer=None) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"[INFO] 체크포인트 로드: {path} (epoch {ckpt.get('epoch', '?')})")
    return ckpt


# ──────────────────────────────────────────────
# 메인 학습 루프
# ──────────────────────────────────────────────
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")
    print(f"[INFO] 설정: {CFG}\n")

    # ── 데이터 준비 ──────────────────────────
    pairs = find_pairs(DATA_DIR)
    train_pairs, val_pairs, _ = split_pairs(
        pairs,
        val_ratio  = CFG["val_ratio"],
        test_ratio = CFG["test_ratio"],
        seed       = CFG["seed"],
    )

    train_ds = VowelWindowDataset(
        train_pairs,
        window_size = CFG["window_size"],
        stride      = CFG["stride"],
        augment     = True,
    )
    val_ds = VowelWindowDataset(
        val_pairs,
        window_size = CFG["window_size"],
        stride      = CFG["stride"],
        augment     = False,
    )

    # ── 클래스 불균형 처리 ───────────────────
    class_weights = compute_class_weights(train_ds, device) if CFG["use_class_weight"] else None

    if CFG["use_weighted_sampler"]:
        sample_weights = compute_sample_weights(train_ds)
        sampler = WeightedRandomSampler(
            weights     = sample_weights,
            num_samples = len(sample_weights),
            replacement = True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size  = CFG["batch_size"],
            sampler     = sampler,
            num_workers = CFG["num_workers"],
            pin_memory  = True,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size  = CFG["batch_size"],
            shuffle     = True,
            num_workers = CFG["num_workers"],
            pin_memory  = True,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size  = CFG["batch_size"],
        shuffle     = False,
        num_workers = CFG["num_workers"],
        pin_memory  = True,
    )

    print(f"\n[INFO] train: {len(train_ds):,}  val: {len(val_ds):,}")

    # ── 모델 / 손실 / 옵티마이저 ─────────────
    model = VowelClassifier(
        d_model            = CFG["d_model"],
        nhead              = CFG["nhead"],
        num_encoder_layers = CFG["num_encoder_layers"],
        dim_feedforward    = CFG["dim_feedforward"],
        dropout            = CFG["dropout"],
        num_classes        = CFG["num_classes"],
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(
        model.parameters(),
        lr           = CFG["learning_rate"],
        weight_decay = CFG["weight_decay"],
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs = CFG["warmup_epochs"],
        total_epochs  = CFG["epochs"],
        base_lr       = CFG["learning_rate"],
    )

    # ── 학습 루프 ─────────────────────────────
    best_val_loss  = float("inf")
    no_improve     = 0
    history        = []

    print("\n" + "="*60)
    print("학습 시작")
    print("="*60)

    for epoch in range(1, CFG["epochs"] + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, is_train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, device, is_train=False
        )

        scheduler.step()
        lr = scheduler.get_lr()
        elapsed = time.time() - t0

        # 로그
        print(
            f"Epoch {epoch:3d}/{CFG['epochs']} | "
            f"lr={lr:.2e} | "
            f"train loss={train_loss:.4f} acc={train_acc:.1f}% | "
            f"val loss={val_loss:.4f} acc={val_acc:.1f}% | "
            f"{elapsed:.1f}s"
        )

        history.append({
            "epoch"      : epoch,
            "train_loss" : train_loss,
            "train_acc"  : train_acc,
            "val_loss"   : val_loss,
            "val_acc"    : val_acc,
            "lr"         : lr,
        })

        # ── 체크포인트 저장 ───────────────────
        ckpt_state = {
            "epoch"          : epoch,
            "model_state"    : model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss"       : val_loss,
            "val_acc"        : val_acc,
            "cfg"            : CFG,
        }

        # latest 항상 저장
        save_checkpoint(ckpt_state, f"{CHECKPOINT_DIR}/latest.pt")

        # best 갱신
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            save_checkpoint(ckpt_state, f"{CHECKPOINT_DIR}/best.pt")
            print(f"  ★ best 갱신 (val_loss={best_val_loss:.4f})")
        else:
            no_improve += 1
            print(f"  (개선 없음 {no_improve}/{CFG['early_stop']})")

        # Early stopping
        if no_improve >= CFG["early_stop"]:
            print(f"\n[INFO] Early stopping at epoch {epoch}")
            break

    # ── 학습 완료 요약 ────────────────────────
    print("\n" + "="*60)
    print("학습 완료")
    print(f"  best val_loss : {best_val_loss:.4f}")
    best_epoch = min(history, key=lambda x: x["val_loss"])
    print(f"  best epoch    : {best_epoch['epoch']}")
    print(f"  best val_acc  : {best_epoch['val_acc']:.1f}%")
    print(f"  체크포인트    : {CHECKPOINT_DIR}/best.pt")
    print("="*60)

    return history


# ──────────────────────────────────────────────
# 실행
# ──────────────────────────────────────────────
if __name__ == "__main__":
    train()
