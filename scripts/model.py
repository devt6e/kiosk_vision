# model.py
# 모음 분류 립리딩 모델
# Input  : (B, 1, 7, 64, 64)  [B, C, T, H, W]
# Output : (B, 8)              [B, num_classes]

import torch
import torch.nn as nn


# ──────────────────────────────────────────────
# 1. Visual Frontend (Conv3d)
# ──────────────────────────────────────────────
class VisualFrontend(nn.Module):
    """
    Conv3d 3단으로 시공간 특징 추출.

    입력  : (B, 1,   7, 64, 64)
    출력  : (B, 128, 7,  8,  8)

    시간 축(T)은 유지 — Transformer에 7개 토큰으로 전달
    공간 축(H,W)은 64 → 8로 압축
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv3d(
                in_channels, 32,
                kernel_size=(3, 3, 3),
                stride=(1, 2, 2),       # 공간만 절반 압축: 64→32
                padding=(1, 1, 1),
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv3d(
                32, 64,
                kernel_size=(3, 3, 3),
                stride=(1, 2, 2),       # 32→16
                padding=(1, 1, 1),
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.layer3 = nn.Sequential(
            nn.Conv3d(
                64, 128,
                kernel_size=(3, 3, 3),
                stride=(1, 2, 2),       # 16→8
                padding=(1, 1, 1),
            ),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 7, 64, 64)
        x = self.layer1(x)   # (B, 32,  7, 32, 32)
        x = self.layer2(x)   # (B, 64,  7, 16, 16)
        x = self.layer3(x)   # (B, 128, 7,  8,  8)
        return x


# ──────────────────────────────────────────────
# 2. Spatial Pooling + Projection
# ──────────────────────────────────────────────
class SpatialPool(nn.Module):
    """
    공간 차원(H, W) → 평균 풀링으로 제거.
    (B, 128, 7, 8, 8) → (B, 7, 512)

    Transformer 입력 형태로 변환:
    - T=7 개의 토큰
    - 각 토큰 차원 = d_model
    """

    def __init__(self, in_channels: int = 128, d_model: int = 512):
        super().__init__()
        self.pool    = nn.AdaptiveAvgPool3d((None, 1, 1))  # (B, C, T, 1, 1)
        self.project = nn.Linear(in_channels, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 128, 7, 8, 8)
        x = self.pool(x)                    # (B, 128, 7, 1, 1)
        x = x.squeeze(-1).squeeze(-1)       # (B, 128, 7)
        x = x.permute(0, 2, 1)             # (B, 7, 128)
        x = self.project(x)                 # (B, 7, 512)
        return x


# ──────────────────────────────────────────────
# 3. Visual Transformer Encoder
# ──────────────────────────────────────────────
class VisualTransformerEncoder(nn.Module):
    """
    Transformer Encoder로 7개 프레임 토큰 간 관계 학습.

    입력  : (B, 7, d_model)
    출력  : (B, d_model)   ← 시간 평균 풀링 후
    """

    def __init__(
        self,
        d_model            : int   = 512,
        nhead              : int   = 8,
        num_encoder_layers : int   = 6,
        dim_feedforward    : int   = 2048,
        dropout            : float = 0.1,
    ):
        super().__init__()

        # 위치 인코딩 (학습 가능, 7개 토큰 고정)
        self.pos_embedding = nn.Parameter(torch.randn(1, 7, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
            batch_first     = True,   # (B, T, d_model) 형식 사용
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers = num_encoder_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 7, d_model)
        x = x + self.pos_embedding              # 위치 정보 추가
        x = self.encoder(x)                     # (B, 7, d_model)
        x = x.mean(dim=1)                       # (B, d_model) 시간 평균
        return x


# ──────────────────────────────────────────────
# 4. Classification Head
# ──────────────────────────────────────────────
class ClassificationHead(nn.Module):
    """
    (B, d_model) → (B, num_classes)

    Dropout → Linear → (선택) 중간 레이어 → Linear
    """

    def __init__(
        self,
        d_model     : int   = 512,
        num_classes : int   = 8,
        dropout     : float = 0.3,
    ):
        super().__init__()

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)   # (B, num_classes)


# ──────────────────────────────────────────────
# 5. 전체 모델
# ──────────────────────────────────────────────
class VowelClassifier(nn.Module):
    """
    립리딩 기반 한국어 모음 분류 모델.

    구조:
        VisualFrontend (Conv3d 3단)
        → SpatialPool + Linear Projection
        → VisualTransformerEncoder
        → ClassificationHead
        → 8클래스 logits

    사용 예시:
        model = VowelClassifier()
        x = torch.randn(32, 1, 7, 64, 64)
        logits = model(x)   # (32, 8)
    """

    def __init__(
        self,
        in_channels        : int   = 1,
        d_model            : int   = 512,
        nhead              : int   = 8,
        num_encoder_layers : int   = 6,
        dim_feedforward    : int   = 2048,
        dropout            : float = 0.1,
        num_classes        : int   = 8,
    ):
        super().__init__()

        self.frontend  = VisualFrontend(in_channels)
        self.pool      = SpatialPool(in_channels=128, d_model=d_model)
        self.encoder   = VisualTransformerEncoder(
            d_model, nhead, num_encoder_layers, dim_feedforward, dropout
        )
        self.head      = ClassificationHead(d_model, num_classes, dropout=0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 7, 64, 64)
        x = self.frontend(x)    # (B, 128, 7, 8, 8)
        x = self.pool(x)        # (B, 7, 512)
        x = self.encoder(x)     # (B, 512)
        x = self.head(x)        # (B, 8)
        return x

    def count_params(self) -> dict:
        """레이어별 파라미터 수 반환"""
        total = sum(p.numel() for p in self.parameters())
        parts = {
            "frontend" : sum(p.numel() for p in self.frontend.parameters()),
            "pool"     : sum(p.numel() for p in self.pool.parameters()),
            "encoder"  : sum(p.numel() for p in self.encoder.parameters()),
            "head"     : sum(p.numel() for p in self.head.parameters()),
        }
        parts["total"] = total
        return parts


# ──────────────────────────────────────────────
# 테스트
# ──────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")

    model = VowelClassifier().to(device)

    # 파라미터 수 확인
    params = model.count_params()
    print("=== 파라미터 수 ===")
    for name, count in params.items():
        print(f"  {name:12s}: {count:,}")

    # Forward pass 확인
    print("\n=== Forward Pass ===")
    x = torch.randn(32, 1, 7, 64, 64).to(device)
    print(f"  입력  : {x.shape}")

    with torch.no_grad():
        logits = model(x)

    print(f"  출력  : {logits.shape}")   # (32, 8)
    print(f"  정상 여부: {logits.shape == torch.Size([32, 8])}")

    # 중간 레이어 shape 확인
    print("\n=== 중간 Shape ===")
    x = torch.randn(4, 1, 7, 64, 64).to(device)
    with torch.no_grad():
        f = model.frontend(x)
        print(f"  Frontend 출력  : {f.shape}")   # (4, 128, 7, 8, 8)
        p = model.pool(f)
        print(f"  SpatialPool 출력: {p.shape}")  # (4, 7, 512)
        e = model.encoder(p)
        print(f"  Encoder 출력   : {e.shape}")   # (4, 512)
        out = model.head(e)
        print(f"  Head 출력      : {out.shape}")  # (4, 8)
