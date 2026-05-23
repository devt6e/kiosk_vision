# vocab.py
# 한국어 텍스트 → 모음 시퀀스 → 8클래스 인덱스 변환

from config import VOWEL_CLASSES, VOWEL_CLASS_NAMES, VOCAB_SIZE

# ──────────────────────────────────────────────
# 한국어 유니코드 상수
# ──────────────────────────────────────────────
# 한글 음절: 가(0xAC00) ~ 힣(0xD7A3)
# 음절 = 초성(19) × 중성(21) × 종성(28) + 0xAC00
HANGUL_START = 0xAC00
HANGUL_END   = 0xD7A3

# 중성(모음) 21개 — 유니코드 분해 순서
JUNGSEONG = [
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ",
    "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ", "ㅚ",
    "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ",
    "ㅡ", "ㅢ", "ㅣ",
]

# 복합 모음 → 단순 모음 매핑
# 입술 모양 기준으로 구성 요소 중 지배적인 모음으로 단순화
COMPOUND_VOWEL_MAP = {
    "ㅘ": "ㅗ",   # ㅗ + ㅏ  → ㅗ 입술 모양 지배
    "ㅙ": "ㅗ",   # ㅗ + ㅐ  → ㅗ 입술 모양 지배
    "ㅚ": "ㅗ",   # ㅗ + ㅣ  → ㅗ 입술 모양 지배
    "ㅝ": "ㅜ",   # ㅜ + ㅓ  → ㅜ 입술 모양 지배
    "ㅞ": "ㅜ",   # ㅜ + ㅔ  → ㅜ 입술 모양 지배
    "ㅟ": "ㅜ",   # ㅜ + ㅣ  → ㅜ 입술 모양 지배
    "ㅢ": "ㅡ",   # ㅡ + ㅣ  → ㅡ 입술 모양 지배
}


class VowelVocab:
    """
    한국어 텍스트를 모음 클래스 인덱스 시퀀스로 변환.

    CTC 구조:
        blank = 0
        모음 클래스 = 1~8
        VOCAB_SIZE = 9

    사용 예시:
        vocab = VowelVocab()
        indices = vocab.text_to_indices("안녕하세요")
        # → [1, 5, 1, 7, 2]  (ㅏ, ㅣ, ㅏ, ㅔ, ㅛ 에 해당하는 클래스)
        decoded = vocab.indices_to_names(indices)
        # → ['ㅏ/ㅑ', 'ㅣ/ㅖ/ㅒ', 'ㅏ/ㅑ', 'ㅔ/ㅐ', 'ㅗ/ㅛ']
    """

    BLANK_IDX     = 0
    SILENCE_IDX   = 8

    def __init__(self):
        # 모음 → 클래스 인덱스 역방향 매핑 생성
        self._vowel_to_idx: dict[str, int] = {}
        for class_idx, vowels in VOWEL_CLASSES.items():
            if vowels == ["침묵"]:
                continue
            for v in vowels:
                self._vowel_to_idx[v] = class_idx

        # 복합 모음도 등록
        for compound, simple in COMPOUND_VOWEL_MAP.items():
            if simple in self._vowel_to_idx:
                self._vowel_to_idx[compound] = self._vowel_to_idx[simple]

        self.vocab_size    = VOCAB_SIZE
        self.class_names   = VOWEL_CLASS_NAMES

    # ──────────────────────────────────────────
    # 한글 음절 분해
    # ──────────────────────────────────────────
    @staticmethod
    def decompose_syllable(char: str) -> dict | None:
        """
        한글 음절 1자 → 초성/중성/종성 분해.
        한글이 아니면 None 반환.
        """
        code = ord(char)
        if not (HANGUL_START <= code <= HANGUL_END):
            return None

        offset    = code - HANGUL_START
        jongseong = offset % 28
        offset    //= 28
        jungseong = offset % 21
        choseong  = offset // 21

        return {
            "choseong":  choseong,
            "jungseong": JUNGSEONG[jungseong],
            "jongseong": jongseong,  # 0 = 받침 없음
        }

    def extract_vowel(self, char: str) -> str | None:
        """
        한글 음절 1자 → 중성(모음) 1개 추출.
        한글이 아니면 None.
        """
        result = self.decompose_syllable(char)
        if result is None:
            return None
        return result["jungseong"]

    # ──────────────────────────────────────────
    # 모음 → 클래스 인덱스
    # ──────────────────────────────────────────
    def vowel_to_idx(self, vowel: str) -> int:
        """
        모음 문자 → 클래스 인덱스.
        미등록 모음은 silence(8) 반환.
        """
        return self._vowel_to_idx.get(vowel, self.SILENCE_IDX)

    # ──────────────────────────────────────────
    # 텍스트 → 인덱스 시퀀스 (핵심 메서드)
    # ──────────────────────────────────────────
    def text_to_indices(self, text: str) -> list[int]:
        """
        한국어 문장 → 모음 클래스 인덱스 리스트.

        - 한글 음절만 처리 (공백, 숫자, 특수문자 무시)
        - 복합 모음은 단순 모음으로 먼저 변환
        - 같은 클래스가 연속되면 중복 제거 (CTC 학습 타겟용)

        예시:
            "안녕" → [ㅏ, ㅕ] → [1, 6]
        """
        indices = []
        for char in text:
            vowel = self.extract_vowel(char)
            if vowel is None:
                continue  # 한글 아닌 문자 스킵

            # 복합 모음 → 단순 모음
            vowel = COMPOUND_VOWEL_MAP.get(vowel, vowel)
            idx   = self.vowel_to_idx(vowel)

            # 연속 중복 제거 (CTC 타겟은 collapse된 형태)
            if not indices or indices[-1] != idx:
                indices.append(idx)

        return indices

    def text_to_indices_raw(self, text: str) -> list[int]:
        """
        중복 제거 없이 음절 순서 그대로 반환.
        라벨 분석, 디버깅용.
        """
        indices = []
        for char in text:
            vowel = self.extract_vowel(char)
            if vowel is None:
                continue
            vowel = COMPOUND_VOWEL_MAP.get(vowel, vowel)
            indices.append(self.vowel_to_idx(vowel))
        return indices

    # ──────────────────────────────────────────
    # 인덱스 → 클래스명
    # ──────────────────────────────────────────
    def indices_to_names(self, indices: list[int]) -> list[str]:
        """인덱스 리스트 → 클래스명 리스트 (출력/디버깅용)"""
        return [self.class_names.get(i, "?") for i in indices]

    def idx_to_name(self, idx: int) -> str:
        return self.class_names.get(idx, "?")

    # ──────────────────────────────────────────
    # 정보 출력
    # ──────────────────────────────────────────
    def __repr__(self):
        lines = ["VowelVocab("]
        lines.append(f"  vocab_size = {self.vocab_size}")
        lines.append(f"  blank_idx  = {self.BLANK_IDX}")
        for idx, name in self.class_names.items():
            lines.append(f"  [{idx}] {name}")
        lines.append(")")
        return "\n".join(lines)


# ──────────────────────────────────────────────
# 테스트
# ──────────────────────────────────────────────
if __name__ == "__main__":
    vocab = VowelVocab()
    print(vocab)
    print()

    test_cases = [
        "안녕하세요",
        "아메리카노 한 잔 주세요",
        "카페라떼",
        "주문할게요",
    ]

    for text in test_cases:
        indices = vocab.text_to_indices(text)
        raw     = vocab.text_to_indices_raw(text)
        names   = vocab.indices_to_names(indices)

        print(f"입력  : {text}")
        print(f"raw   : {raw}  {vocab.indices_to_names(raw)}")
        print(f"CTC용 : {indices}  {names}")
        print()

    # 복합 모음 확인
    print("=== 복합 모음 매핑 확인 ===")
    for compound, simple in COMPOUND_VOWEL_MAP.items():
        idx = vocab.vowel_to_idx(compound)
        print(f"  {compound} → {simple} → [{idx}] {vocab.idx_to_name(idx)}")
