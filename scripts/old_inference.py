# inference.py
# 발화 단위 모음 분류 추론 파이프라인 (B방식)
#
# 흐름:
#   카메라 → MediaPipe 입술 크롭 → 프레임 버퍼 수집
#   → 발화 감지 (침묵 기준) → 슬라이딩 윈도우 추론
#   → 모음 시퀀스 출력 → 키오스크 콜백

import os
import time
import argparse
import collections
import threading
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from model import VowelClassifier
from dataset import LABEL_TO_IDX, IDX_TO_LABEL, NUM_CLASSES

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
CHECKPOINT_PATH = os.path.expanduser("~/lip_reading/checkpoints/best.pt")
MODEL_DIR       = os.path.expanduser("~/lip_reading/models/")
LANDMARKER_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")

INF_CFG = {
    # 슬라이딩 윈도우 (학습과 동일)
    "window_size"        : 7,
    "stride"             : 2,

    # 발화 감지 (모션 기반)
    "silence_idx"        : 7,
    "silence_threshold"  : 0.45,
    "motion_threshold"   : 5.0,          # 0=침묵, 13~41=발화 → 중간값
    "silence_duration"   : 0.8,          # 마지막 모션 후 N초 경과 → 침묵
    "min_silence_frames" : 8,            # 발화 종료 판단 연속 침묵 프레임
    "min_speech_frames"  : 10,           # 유효 발화 최소 프레임 수
    "max_speech_frames"  : 150,          # 최대 발화 프레임 (초과 시 강제 종료)

    # 카메라
    "camera_index"       : 0,
    "frame_size"         : (64, 64),
    "display"            : True,

    # 모델
    "d_model"            : 512,
    "nhead"              : 8,
    "num_encoder_layers" : 4,
    "dim_feedforward"    : 2048,
    "dropout"            : 0.1,
    "num_classes"        : 8,
}

# ──────────────────────────────────────────────
# 1. 입술 감지 (MediaPipe Face Mesh)
# ──────────────────────────────────────────────
class LipDetector:
    """
    MediaPipe Face Mesh로 입술 영역 감지 및 크롭.

    Tasks API (mediapipe >= 0.10) 사용.
    landmarker.task 파일 필요.
    """

    # 입술 외곽 랜드마크 인덱스
    LIP_LANDMARKS = [
        61, 146, 91, 181, 84, 17, 314, 405,
        321, 375, 291, 409, 270, 269, 267, 0,
        37, 39, 40, 185,
    ]

    def __init__(self, model_path: str):
        self.available = False
        self._init_mediapipe(model_path)

    def _init_mediapipe(self, model_path: str):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = mp_vision.FaceLandmarkerOptions(
                base_options       = base_options,
                output_face_blendshapes = False,
                num_faces          = 1,
                min_face_detection_confidence = 0.5,
                min_face_presence_confidence  = 0.5,
                min_tracking_confidence       = 0.5,
            )
            self.detector  = mp_vision.FaceLandmarker.create_from_options(options)
            self.mp_image  = mp.Image
            self.mp_format = mp.ImageFormat.SRGB
            self.available = True
            print("[INFO] MediaPipe LipDetector 초기화 완료")

        except Exception as e:
            print(f"[WARN] MediaPipe 초기화 실패: {e}")
            print("[WARN] Fallback: 고정 중앙 크롭 사용")

    def crop(self, frame_bgr: np.ndarray, out_size: tuple = (64, 64)) -> np.ndarray | None:
        """
        BGR 프레임 → 입술 크롭 (out_size).
        감지 실패 시 None 반환.
        """
        if self.available:
            return self._crop_mediapipe(frame_bgr, out_size)
        else:
            return self._crop_center(frame_bgr, out_size)

    def _crop_mediapipe(self, frame_bgr: np.ndarray, out_size: tuple) -> np.ndarray | None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img    = self.mp_image(image_format=self.mp_format, data=frame_rgb)
        result    = self.detector.detect(mp_img)

        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        h, w      = frame_bgr.shape[:2]

        # 입술 랜드마크 좌표 수집
        xs = [int(landmarks[i].x * w) for i in self.LIP_LANDMARKS]
        ys = [int(landmarks[i].y * h) for i in self.LIP_LANDMARKS]

        # 바운딩 박스 + 여백
        x1 = max(0, min(xs) - 10)
        y1 = max(0, min(ys) - 10)
        x2 = min(w, max(xs) + 10)
        y2 = min(h, max(ys) + 10)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame_bgr[y1:y2, x1:x2]
        crop = cv2.resize(crop, out_size)
        return crop

    def _crop_center(self, frame_bgr: np.ndarray, out_size: tuple) -> np.ndarray:
        """MediaPipe 없을 때 하단 중앙 고정 크롭 (fallback)"""
        h, w   = frame_bgr.shape[:2]
        cx, cy = w // 2, int(h * 0.75)
        half   = min(w, h) // 6
        x1, y1 = max(0, cx - half), max(0, cy - half)
        x2, y2 = min(w, cx + half), min(h, cy + half)
        crop   = frame_bgr[y1:y2, x1:x2]
        return cv2.resize(crop, out_size)


# ──────────────────────────────────────────────
# 2. 모음 추론기
# ──────────────────────────────────────────────
class VowelInferencer:
    """
    학습된 모델로 프레임 버퍼 → 모음 시퀀스 추론.
    슬라이딩 윈도우(window=7, stride=2) 적용.
    """

    def __init__(self, checkpoint_path: str, device: torch.device):
        self.device      = device
        self.window_size = INF_CFG["window_size"]
        self.stride      = INF_CFG["stride"]
        self.half        = self.window_size // 2
        self.silence_idx = INF_CFG["silence_idx"]
        self.silence_thr = INF_CFG["silence_threshold"]
        self.model       = self._load_model(checkpoint_path)

    def _load_model(self, path: str) -> VowelClassifier:
        model = VowelClassifier(
            d_model            = INF_CFG["d_model"],
            nhead              = INF_CFG["nhead"],
            num_encoder_layers = INF_CFG["num_encoder_layers"],
            dim_feedforward    = INF_CFG["dim_feedforward"],
            dropout            = INF_CFG["dropout"],
            num_classes        = INF_CFG["num_classes"],
        ).to(self.device)

        ckpt = torch.load(path, map_location=self.device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        print(f"[INFO] 모델 로드 완료 (epoch={ckpt.get('epoch','?')}, "
              f"val_acc={ckpt.get('val_acc', 0):.1f}%)")
        return model

    def preprocess_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """BGR (64,64,3) → 그레이스케일 float32 (64,64)"""
        gray = (
            0.299 * frame_bgr[:, :, 2].astype(np.float32) +
            0.587 * frame_bgr[:, :, 1].astype(np.float32) +
            0.114 * frame_bgr[:, :, 0].astype(np.float32)
        ) / 255.0
        return gray   # (64, 64)

    def infer(self, frames: list[np.ndarray]) -> list[dict]:
        """
        프레임 리스트 → 윈도우별 예측 결과 리스트.

        frames : list of (64,64,3) uint8 BGR
        반환   : [{"label": str, "idx": int, "prob": float}, ...]
                  윈도우 개수만큼 반환
        """
        if len(frames) < self.window_size:
            return []

        # 전처리
        gray_frames = np.stack(
            [self.preprocess_frame(f) for f in frames]
        )   # (T, 64, 64)

        T       = len(gray_frames)
        results = []

        # 슬라이딩 윈도우 배치 구성
        windows = []
        centers = range(self.half, T - self.half, self.stride)
        for c in centers:
            win = gray_frames[c - self.half: c + self.half + 1]  # (7, 64, 64)
            windows.append(win)

        if not windows:
            return []

        # 배치 텐서 변환: (N, 1, 7, 64, 64)
        batch = torch.from_numpy(
            np.stack(windows)[:, np.newaxis]
        ).float().to(self.device)

        with torch.no_grad():
            logits = self.model(batch)              # (N, 8)
            probs  = F.softmax(logits, dim=-1)      # (N, 8)

        for i in range(len(windows)):
            prob_arr  = probs[i].cpu().numpy()
            pred_idx  = int(prob_arr.argmax())
            pred_prob = float(prob_arr[pred_idx])
            results.append({
                "idx"  : pred_idx,
                "label": IDX_TO_LABEL[pred_idx],
                "prob" : pred_prob,
                "probs": prob_arr,
            })

        return results

    def decode_sequence(self, results: list[dict]) -> list[str]:
        """
        윈도우별 예측 → 모음 시퀀스 (연속 중복 제거, 침묵 제거).

        예: [A, A, SILENCE, O, O, O] → ['A', 'O']
        """
        seq = []
        for r in results:
            label = r["label"]
            if label == "SILENCE":
                continue
            if not seq or seq[-1] != label:
                seq.append(label)
        return seq


# ──────────────────────────────────────────────
# 3. 발화 감지기
# ──────────────────────────────────────────────
class UtteranceDetector:
    """
    모션 기반 발화 감지 + 모음 분류 추론.

    침묵 감지: 프레임 간 픽셀 차이(모션) 기반
      → 입술 움직임 없음 = 침묵
      → 모델 정확도와 무관하게 안정적으로 동작

    상태:
        WAITING  → pre-buffer 채우며 모션 확인 대기
        SPEAKING → 발화 중 (프레임 수집)
        ENDING   → 침묵 확인 중
    """

    WAITING  = "WAITING"
    SPEAKING = "SPEAKING"
    ENDING   = "ENDING"

    def __init__(self, inferencer: VowelInferencer):
        self.inferencer         = inferencer
        self.state              = self.WAITING
        self.frame_buffer: list = []
        self.pre_buffer: list   = []
        self.silence_count      = 0
        self.speech_trigger     = 0
        self.min_silence        = INF_CFG["min_silence_frames"]
        self.min_speech         = INF_CFG["min_speech_frames"]
        self.max_speech         = INF_CFG["max_speech_frames"]
        self.window_size        = inferencer.window_size
        self.speech_trigger_thr = 3

        # 모션 기반 침묵 감지
        self.motion_threshold   = INF_CFG["motion_threshold"]
        self.silence_duration   = INF_CFG["silence_duration"]   # 초
        self.prev_gray          = None
        self.last_raw_motion    = 0.0
        self.last_motion_time   = time.time()   # 마지막 모션 감지 시각

    def _calc_motion(self, frame_bgr: np.ndarray) -> float:
        """이전 프레임과의 픽셀 차이로 모션 계산"""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0
        diff             = np.abs(gray - self.prev_gray)
        motion           = float(diff.mean())
        self.prev_gray   = gray
        self.last_raw_motion = motion
        return motion

    def _is_silence(self, frame_bgr: np.ndarray) -> tuple[bool, float]:
        """
        타이머 기반 침묵 판단.
        motion > threshold 가 감지되면 타이머 리셋.
        마지막 모션 감지 후 silence_duration 초 경과 → 침묵.
        """
        motion = self._calc_motion(frame_bgr)

        if motion > self.motion_threshold:
            self.last_motion_time = time.time()   # 타이머 리셋

        elapsed = time.time() - self.last_motion_time
        is_sil  = elapsed >= self.silence_duration

        return is_sil, motion

    def _is_silence_last(self) -> tuple[bool, float]:
        """디버그용 — 현재 침묵 여부와 마지막 모션값 반환"""
        elapsed = time.time() - self.last_motion_time
        return elapsed >= self.silence_duration, self.last_raw_motion

    def push_frame(self, frame_bgr: np.ndarray, debug: bool = False) -> list[str] | None:

        is_sil, motion_val = self._is_silence(frame_bgr)

        # 0.5초마다 모션값 출력
        if debug:
            now = time.time()
            if not hasattr(self, '_last_debug_time'):
                self._last_debug_time = 0.0
            if now - self._last_debug_time >= 0.5:
                tag   = "SILENCE" if is_sil else "MOTION"
                extra = ""
                if self.state == self.WAITING:
                    extra = f" trigger={self.speech_trigger}/{self.speech_trigger_thr}"
                elif self.state == self.SPEAKING:
                    extra = f" frames={len(self.frame_buffer)}"
                elif self.state == self.ENDING:
                    extra = f" frames={len(self.frame_buffer)} silence={self.silence_count}/{self.min_silence}"
                print(f"[{self.state:8s}] motion={motion_val:.2f} {tag}{extra}")
                self._last_debug_time = now

        if self.state == self.WAITING:
            self.pre_buffer.append(frame_bgr)
            if len(self.pre_buffer) > self.window_size * 2:
                self.pre_buffer.pop(0)

            if not is_sil:
                self.speech_trigger += 1
                if self.speech_trigger >= self.speech_trigger_thr:
                    self.state          = self.SPEAKING
                    self.frame_buffer   = self.pre_buffer.copy()
                    self.pre_buffer     = []
                    self.speech_trigger = 0
                    self.silence_count  = 0
                    print(f"[INFO] 발화 시작 감지 (motion={motion_val:.2f})")
            else:
                self.speech_trigger = 0
            return None

        elif self.state == self.SPEAKING:
            self.frame_buffer.append(frame_bgr)

            if len(self.frame_buffer) >= self.max_speech:
                print(f"[INFO] 최대 프레임 초과 → 강제 종료")
                return self._finalize()

            if is_sil:
                self.state         = self.ENDING
                self.silence_count = 1
            return None

        elif self.state == self.ENDING:
            self.frame_buffer.append(frame_bgr)

            if is_sil:
                self.silence_count += 1
                if self.silence_count >= self.min_silence:
                    return self._finalize()
            else:
                self.state         = self.SPEAKING
                self.silence_count = 0
            return None

        return None

    def _finalize(self) -> list[str] | None:
        frames = self.frame_buffer
        self.state          = self.WAITING
        self.frame_buffer   = []
        self.pre_buffer     = []
        self.silence_count  = 0
        self.speech_trigger = 0
        self.prev_gray      = None
        self.last_raw_motion = 0.0
        self.last_motion_time = time.time()

        if len(frames) < self.min_speech:
            print(f"[INFO] 발화 너무 짧음 ({len(frames)}프레임), 무시")
            return None

        results = self.inferencer.infer(frames)
        seq     = self.inferencer.decode_sequence(results)
        print(f"[INFO] 발화 완료: {len(frames)}프레임 → {seq}")
        return seq

    @property
    def current_state(self) -> str:
        return self.state


# ──────────────────────────────────────────────
# 4. 메인 추론 루프
# ──────────────────────────────────────────────
def run_inference(on_result=None, use_socket: bool = False, host: str = None, debug: bool = False):
    """
    실시간 카메라 추론 루프.

    on_result : 모음 시퀀스 결과를 받을 콜백 함수
                signature: on_result(vowel_sequence: list[str])
                None이면 콘솔 출력만
    use_socket: WSL 소켓 카메라 사용 여부
    host      : 소켓 호스트 IP
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    # ── 모델 로드 ─────────────────────────────
    inferencer = VowelInferencer(CHECKPOINT_PATH, device)
    detector   = LipDetector(LANDMARKER_PATH)
    utterance  = UtteranceDetector(inferencer)

    # ── 카메라 초기화 ─────────────────────────
    if use_socket:
        cap = _init_socket_camera(host)
    else:
        cap = cv2.VideoCapture(INF_CFG["camera_index"])

    if not cap.isOpened():
        print("[ERROR] 카메라를 열 수 없습니다.")
        return

    print("[INFO] 추론 시작. 'q' 키로 종료.")
    print("[INFO] 발화하면 모음 시퀀스가 출력됩니다.\n")

    display = INF_CFG["display"]

    # 0.5초마다 모션값 출력 (별도 스레드 — 프레임 수신과 무관하게 동작)
    if debug:
        def _motion_printer():
            peak = 0.0
            while True:
                time.sleep(0.5)
                is_sil, motion_val = utterance._is_silence_last()
                elapsed = time.time() - utterance.last_motion_time
                peak    = max(peak, motion_val)
                tag     = "SILENCE" if is_sil else "MOTION "
                print(f"[MOTION] now={motion_val:.2f}  peak={peak:.2f}  "
                      f"silent={elapsed:.1f}s  {tag}  state={utterance.current_state}")
        threading.Thread(target=_motion_printer, daemon=True).start()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.03)
                continue

            # 소켓 모드: Windows에서 이미 lip crop 완료 → 그대로 사용
            # 로컬 카메라 모드: MediaPipe로 crop
            if use_socket:
                crop = frame
            else:
                crop = detector.crop(frame, out_size=INF_CFG["frame_size"])
                if crop is None:
                    continue

            # 발화 감지 + 추론
            result = utterance.push_frame(crop, debug=False)

            if result is not None:
                print(f"[결과] 모음 시퀀스: {result}")
                if on_result:
                    on_result(result)

            # 화면 출력
            if display:
                status = utterance.current_state
                color  = {
                    "WAITING" : (100, 100, 100),
                    "SPEAKING": (0, 255, 0),
                    "ENDING"  : (0, 165, 255),
                }.get(status, (255, 255, 255))
                cv2.putText(
                    frame, f"State: {status}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
                )
                cv2.imshow("Vowel Inference", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()
        print("[INFO] 추론 종료")


# ──────────────────────────────────────────────
# 5. 소켓 카메라 (WSL ↔ Windows)
# ──────────────────────────────────────────────
def _init_socket_camera(host: str | None):
    """WSL 환경에서 Windows 카메라를 소켓으로 연결"""
    import socket, struct, threading

    if host is None:
        # WSL 기본 게이트웨이 자동 탐지
        import subprocess
        result = subprocess.run(
            ["ip", "route", "show", "default"],
            capture_output=True, text=True
        )
        host = result.stdout.split()[2]
        print(f"[INFO] 소켓 호스트 자동 탐지: {host}")

    port = 9999
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    print(f"[INFO] 소켓 연결: {host}:{port}")

    class SocketCapture:
        def __init__(self, s):
            self.sock      = s
            self.frame     = None
            self.opened    = True
            self.is_new    = False        # 새 프레임 여부 플래그
            self._lock     = threading.Lock()
            t = threading.Thread(target=self._recv_loop, daemon=True)
            t.start()

        def _recv_exact(self, n: int) -> bytes | None:
            """정확히 n바이트를 받을 때까지 대기"""
            buf = b""
            while len(buf) < n:
                chunk = self.sock.recv(n - len(buf))
                if not chunk:
                    return None
                buf += chunk
            return buf

        def _recv_loop(self):
            while True:
                try:
                    header = self._recv_exact(4)
                    if header is None:
                        print("[INFO] 소켓 연결 종료")
                        self.opened = False
                        break

                    size = struct.unpack(">I", header)[0]
                    if size <= 0 or size > 10 * 1024 * 1024:
                        print(f"[WARN] 비정상 패킷 크기: {size}, 스킵")
                        continue

                    data = self._recv_exact(size)
                    if data is None:
                        print("[INFO] 데이터 수신 중 연결 종료")
                        self.opened = False
                        break

                    arr   = np.frombuffer(data, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        with self._lock:
                            self.frame  = frame
                            self.is_new = True   # 새 프레임 도착 표시

                except Exception as e:
                    print(f"[WARN] 수신 오류: {e}")
                    self.opened = False
                    break

        def read(self) -> tuple[bool, np.ndarray | None]:
            """
            새 프레임이 있을 때만 True 반환.
            같은 프레임을 중복 반환하지 않음.
            """
            with self._lock:
                if not self.is_new or self.frame is None:
                    return False, None   # 새 프레임 없음
                self.is_new = False      # 소비 완료
                return True, self.frame.copy()

        def isOpened(self):
            return self.opened

        def release(self):
            self.sock.close()
            self.opened = False

    return SocketCapture(sock)


# ──────────────────────────────────────────────
# 6. 단일 클립 테스트 (카메라 없이)
# ──────────────────────────────────────────────
def test_with_file(frames_path: str):
    """
    .npy 파일로 추론 테스트 (카메라 없이 동작 확인용).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inferencer = VowelInferencer(CHECKPOINT_PATH, device)

    frames_np = np.load(frames_path)   # (T, 64, 64, 3)
    frames    = [frames_np[i] for i in range(len(frames_np))]

    print(f"[INFO] 파일: {frames_path}")
    print(f"[INFO] 총 프레임: {len(frames)}")

    results = inferencer.infer(frames)
    seq     = inferencer.decode_sequence(results)

    print(f"\n[결과] 윈도우별 예측:")
    for i, r in enumerate(results):
        bar = "█" * int(r["prob"] * 20)
        print(f"  [{i:3d}] {r['label']:8s} {r['prob']:.2f} {bar}")

    print(f"\n[결과] 모음 시퀀스: {seq}")
    return seq


# ──────────────────────────────────────────────
# 실행
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket",    action="store_true", help="WSL 소켓 카메라 사용")
    parser.add_argument("--host",      type=str, default=None, help="소켓 호스트 IP")
    parser.add_argument("--test-file", type=str, default=None, help=".npy 파일로 테스트")
    parser.add_argument("--no-display",action="store_true", help="화면 출력 없음 (headless)")
    parser.add_argument("--debug",     action="store_true", help="프레임별 예측 출력")
    args = parser.parse_args()

    if args.no_display:
        INF_CFG["display"] = False

    if args.test_file:
        test_with_file(args.test_file)
    else:
        run_inference(use_socket=args.socket, host=args.host, debug=args.debug)
