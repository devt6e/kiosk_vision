# inference.py
# 발화 단위 모음 분류 추론 파이프라인

import os
import time
import argparse
import threading
import collections
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
LANDMARKER_PATH = os.path.expanduser("~/lip_reading/models/face_landmarker.task")

INF_CFG = {
    # 모델
    "d_model"            : 512,
    "nhead"              : 8,
    "num_encoder_layers" : 4,
    "dim_feedforward"    : 2048,
    "dropout"            : 0.1,
    "num_classes"        : 8,

    # 슬라이딩 윈도우
    "window_size"        : 7,
    "stride"             : 2,

    # 발화 감지
    "motion_threshold"   : 10.0,   # 이 값 초과 → 발화
    "speech_trigger_thr" : 3,     # N프레임 연속 발화 감지 → SPEAKING 전환
    "silence_duration"   : 0.8,   # 마지막 발화 후 N초 경과 → 침묵 확정
    "min_speech_frames"  : 10,    # 유효 발화 최소 프레임
    "max_speech_frames"  : 150,   # 최대 발화 프레임 (강제 종료)

    # 카메라
    "camera_index"       : 0,
    "frame_size"         : (64, 64),
    "display"            : True,
}


# ──────────────────────────────────────────────
# 1. 입술 감지 (MediaPipe)
# ──────────────────────────────────────────────
LIP_LANDMARKS = [
    61, 146, 91, 181, 84, 17, 314, 405,
    321, 375, 291, 409, 270, 269, 267, 0,
    37, 39, 40, 185,
]

class LipDetector:
    def __init__(self, model_path: str):
        self.available = False
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = mp_vision.FaceLandmarkerOptions(
                base_options                  = base_options,
                output_face_blendshapes       = False,
                num_faces                     = 1,
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
        if self.available:
            return self._crop_mediapipe(frame_bgr, out_size)
        return self._crop_center(frame_bgr, out_size)

    def _crop_mediapipe(self, frame_bgr: np.ndarray, out_size: tuple) -> np.ndarray | None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img    = self.mp_image(image_format=self.mp_format, data=frame_rgb)
        result    = self.detector.detect(mp_img)

        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        h, w      = frame_bgr.shape[:2]
        xs = [int(landmarks[i].x * w) for i in LIP_LANDMARKS]
        ys = [int(landmarks[i].y * h) for i in LIP_LANDMARKS]
        x1 = max(0, min(xs) - 10)
        y1 = max(0, min(ys) - 10)
        x2 = min(w, max(xs) + 10)
        y2 = min(h, max(ys) + 10)

        if x2 <= x1 or y2 <= y1:
            return None

        return cv2.resize(frame_bgr[y1:y2, x1:x2], out_size)

    def _crop_center(self, frame_bgr: np.ndarray, out_size: tuple) -> np.ndarray:
        h, w   = frame_bgr.shape[:2]
        cx, cy = w // 2, int(h * 0.75)
        half   = min(w, h) // 6
        x1, y1 = max(0, cx - half), max(0, cy - half)
        x2, y2 = min(w, cx + half), min(h, cy + half)
        return cv2.resize(frame_bgr[y1:y2, x1:x2], out_size)


# ──────────────────────────────────────────────
# 2. 모음 추론기
# ──────────────────────────────────────────────
class VowelInferencer:
    def __init__(self, checkpoint_path: str, device: torch.device):
        self.device      = device
        self.window_size = INF_CFG["window_size"]
        self.stride      = INF_CFG["stride"]
        self.half        = self.window_size // 2
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

        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        print(f"[INFO] 모델 로드 완료 (epoch={ckpt.get('epoch','?')}, "
              f"val_acc={ckpt.get('val_acc', 0):.1f}%)")
        return model

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        """BGR (64,64,3) → 그레이스케일 float32 (64,64)"""
        return (
            0.299 * frame_bgr[:, :, 2].astype(np.float32) +
            0.587 * frame_bgr[:, :, 1].astype(np.float32) +
            0.114 * frame_bgr[:, :, 0].astype(np.float32)
        ) / 255.0

    def infer(self, frames: list[np.ndarray]) -> list[dict]:
        """프레임 리스트 → 윈도우별 예측 결과"""
        if len(frames) < self.window_size:
            return []

        gray = np.stack([self._preprocess(f) for f in frames])  # (T, 64, 64)
        T    = len(gray)

        windows = []
        for c in range(self.half, T - self.half, self.stride):
            windows.append(gray[c - self.half: c + self.half + 1])

        if not windows:
            return []

        batch = torch.from_numpy(
            np.stack(windows)[:, np.newaxis]
        ).float().to(self.device)   # (N, 1, 7, 64, 64)

        with torch.no_grad():
            probs = F.softmax(self.model(batch), dim=-1)  # (N, 8)

        results = []
        for i in range(len(windows)):
            p        = probs[i].cpu().numpy()
            pred_idx = int(p.argmax())
            results.append({
                "idx"  : pred_idx,
                "label": IDX_TO_LABEL[pred_idx],
                "prob" : float(p[pred_idx]),
                "probs": p,
            })
        return results

    def decode(self, results: list[dict]) -> list[str]:
        """윈도우별 예측 → 침묵 제거 + 중복 제거"""
        seq = []
        for r in results:
            if r["label"] == "SILENCE":
                continue
            if not seq or seq[-1] != r["label"]:
                seq.append(r["label"])
        return seq


# ──────────────────────────────────────────────
# 3. 발화 감지기 (단순화)
# ──────────────────────────────────────────────
class UtteranceDetector:
    """
    발화 감지 로직 (단순화)

    침묵 판단: motion <= threshold
    발화 판단: motion >  threshold

    상태 전환:
        WAITING  → 발화 프레임 N개 연속 → SPEAKING
        SPEAKING → 마지막 발화 후 silence_duration 초 경과 → 종료
    """

    WAITING  = "WAITING"
    SPEAKING = "SPEAKING"

    def __init__(self, inferencer: VowelInferencer):
        self.inferencer        = inferencer
        self.state             = self.WAITING
        self.frame_buffer      = []
        self.pre_buffer        = []
        self.speech_trigger    = 0

        self.motion_thr        = INF_CFG["motion_threshold"]
        self.trigger_thr       = INF_CFG["speech_trigger_thr"]
        self.silence_duration  = INF_CFG["silence_duration"]
        self.min_speech        = INF_CFG["min_speech_frames"]
        self.max_speech        = INF_CFG["max_speech_frames"]

        self.prev_gray         = None
        self.last_motion_val   = 0.0
        self.last_speech_time  = 0.0   # 마지막 발화 프레임 시각

    def _motion(self, frame_bgr: np.ndarray) -> float:
        """프레임 간 픽셀 차이 → 모션값"""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0
        val            = float(np.abs(gray - self.prev_gray).mean())
        self.prev_gray = gray
        self.last_motion_val = val
        return val

    def push_frame(self, frame_bgr: np.ndarray) -> list[str] | None:
        motion   = self._motion(frame_bgr)
        is_speak = motion > self.motion_thr   # 단순 threshold 비교

        if self.state == self.WAITING:
            self.pre_buffer.append(frame_bgr)
            if len(self.pre_buffer) > self.inferencer.window_size * 2:
                self.pre_buffer.pop(0)

            if motion == 0.0:
                # 동일 프레임 → 무시
                return None

            if is_speak:
                self.speech_trigger += 1
                if self.speech_trigger >= self.trigger_thr:
                    # 발화 시작
                    self.state           = self.SPEAKING
                    self.frame_buffer    = self.pre_buffer.copy()
                    self.pre_buffer      = []
                    self.speech_trigger  = 0
                    self.last_speech_time = time.time()
                    print(f"[INFO] 발화 시작 (motion={motion:.2f})")
            else:
                # 침묵 프레임 → trigger 리셋
                self.speech_trigger = 0
            return None

        elif self.state == self.SPEAKING:
            self.frame_buffer.append(frame_bgr)

            if is_speak:
                self.last_speech_time = time.time()

            # 최대 프레임 초과 → 강제 종료
            if len(self.frame_buffer) >= self.max_speech:
                print("[INFO] 최대 프레임 초과 → 강제 종료")
                return self._finalize()

            # 마지막 발화 후 silence_duration 초 경과 → 종료
            elapsed = time.time() - self.last_speech_time
            if elapsed >= self.silence_duration:
                print(f"[INFO] 침묵 {elapsed:.1f}s → 발화 종료")
                return self._finalize()

            return None

        return None

    def _finalize(self) -> list[str] | None:
        frames            = self.frame_buffer
        self.state        = self.WAITING
        self.frame_buffer = []
        self.pre_buffer   = []
        self.speech_trigger = 0
        self.prev_gray    = None
        self.last_motion_val = 0.0
        self.last_speech_time = 0.0

        if len(frames) < self.min_speech:
            print(f"[INFO] 발화 너무 짧음 ({len(frames)}프레임), 무시")
            return None

        results = self.inferencer.infer(frames)
        seq     = self.inferencer.decode(results)
        print(f"[INFO] 발화 완료: {len(frames)}프레임 → {seq}")
        return seq

    @property
    def current_state(self) -> str:
        return self.state


# ──────────────────────────────────────────────
# 4. 소켓 카메라 (WSL ↔ Windows)
# ──────────────────────────────────────────────
def _init_socket_camera(host: str | None):
    import socket, struct

    if host is None:
        import subprocess
        result = subprocess.run(
            ["ip", "route", "show", "default"],
            capture_output=True, text=True
        )
        host = result.stdout.split()[2]
        print(f"[INFO] 소켓 호스트 자동 탐지: {host}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, 9999))
    print(f"[INFO] 소켓 연결: {host}:9999")

    class SocketCapture:
        def __init__(self, s):
            self.sock   = s
            self.frame  = None
            self.is_new = False
            self.opened = True
            self._lock  = threading.Lock()
            threading.Thread(target=self._recv_loop, daemon=True).start()

        def _recv_exact(self, n):
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
                    if not header:
                        self.opened = False
                        break
                    size = __import__('struct').unpack(">I", header)[0]
                    if size <= 0 or size > 10 * 1024 * 1024:
                        continue
                    data = self._recv_exact(size)
                    if not data:
                        self.opened = False
                        break
                    arr   = np.frombuffer(data, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        with self._lock:
                            self.frame  = frame
                            self.is_new = True
                except Exception as e:
                    print(f"[WARN] 수신 오류: {e}")
                    self.opened = False
                    break

        def read(self):
            with self._lock:
                if not self.is_new or self.frame is None:
                    return False, None
                self.is_new = False
                return True, self.frame.copy()

        def isOpened(self):
            return self.opened

        def release(self):
            self.sock.close()
            self.opened = False

    return SocketCapture(sock)


# ──────────────────────────────────────────────
# 5. 파일 기반 테스트
# ──────────────────────────────────────────────
def test_with_file(frames_path: str):
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inferencer = VowelInferencer(CHECKPOINT_PATH, device)
    frames_np  = np.load(frames_path)
    frames     = [frames_np[i] for i in range(len(frames_np))]

    print(f"[INFO] 파일: {frames_path}  총 프레임: {len(frames)}")
    results = inferencer.infer(frames)
    seq     = inferencer.decode(results)

    print(f"\n[결과] 윈도우별 예측 (상위만):")
    for i, r in enumerate(results[:20]):
        bar = "█" * int(r["prob"] * 20)
        print(f"  [{i:3d}] {r['label']:8s} {r['prob']:.2f} {bar}")

    print(f"\n[결과] 모음 시퀀스: {seq}")
    return seq


# ──────────────────────────────────────────────
# 6. 메인 추론 루프
# ──────────────────────────────────────────────
def run_inference(on_result=None, use_socket=False, host=None, debug=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    inferencer = VowelInferencer(CHECKPOINT_PATH, device)
    detector   = LipDetector(LANDMARKER_PATH)
    utterance  = UtteranceDetector(inferencer)

    cap = _init_socket_camera(host) if use_socket else cv2.VideoCapture(INF_CFG["camera_index"])

    if not cap.isOpened():
        print("[ERROR] 카메라를 열 수 없습니다.")
        return

    print("[INFO] 추론 시작. 'q' 키로 종료.")
    print("[INFO] 발화하면 모음 시퀀스가 출력됩니다.\n")

    # 디버그: 0.5초마다 모션값 출력
    if debug:
        def _printer():
            peak = 0.0
            while True:
                time.sleep(0.5)
                val  = utterance.last_motion_val
                peak = max(peak, val)
                elapsed = time.time() - utterance.last_speech_time if utterance.last_speech_time else 0
                tag  = "MOTION " if val > INF_CFG["motion_threshold"] else "SILENCE"
                print(f"[DEBUG] motion={val:.2f}  peak={peak:.2f}  "
                      f"silent={elapsed:.1f}s  {tag}  state={utterance.current_state}")
        threading.Thread(target=_printer, daemon=True).start()

    display = INF_CFG["display"]
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.03)
                continue

            # 소켓: Windows가 이미 crop → 그대로 사용
            # 로컬:  MediaPipe로 crop
            if use_socket:
                crop = frame
            else:
                crop = detector.crop(frame, out_size=INF_CFG["frame_size"])
                if crop is None:
                    continue

            result = utterance.push_frame(crop)

            if result is not None:
                print(f"[결과] {result}")
                if on_result:
                    on_result(result)

            if display:
                color = {"WAITING": (100,100,100), "SPEAKING": (0,255,0)}.get(
                    utterance.current_state, (255,255,255))
                cv2.putText(frame, f"State: {utterance.current_state}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.imshow("Vowel Inference", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()
        print("[INFO] 추론 종료")


# ──────────────────────────────────────────────
# 실행
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket",     action="store_true")
    parser.add_argument("--host",       type=str, default=None)
    parser.add_argument("--test-file",  type=str, default=None)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--debug",      action="store_true")
    args = parser.parse_args()

    if args.no_display:
        INF_CFG["display"] = False

    if args.test_file:
        test_with_file(args.test_file)
    else:
        run_inference(use_socket=args.socket, host=args.host, debug=args.debug)
