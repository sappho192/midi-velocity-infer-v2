# Gradio Demo Plan: MIDI Velocity Inference

## 개요

Merged model + Gradio 기반 Python demo. 유저가 MIDI 파일을 업로드하면 각 노트의 Velocity를 예측하여 시각화 및 다운로드를 제공한다.

## 배포 환경

- **타겟**: HuggingFace Spaces (CPU Free Tier, RAM 16GB)
- **런타임**: OnnxRuntime (CPUExecutionProvider), PyTorch 의존성 없음
- **코드 위치**: `v3/demo/`에서 개발, HF Space 배포 시 필요 파일만 추출

## UI 레이아웃

좌우 분할 (Left: 설정 패널, Right: 결과 표시):

```
┌────────────┬──────────────────┐
│ [Upload]   │ [Piano Roll]     │
│ Model:[__] │ [Velocity Bars]  │
│ Ctrl:[___] │                  │
│ Temp:[___] │                  │
│ [Generate] │ [Audio Player]   │
│            │ [Download MIDI]  │
└────────────┴──────────────────┘
```

## 모델 선택

3개 모델을 별칭 + 기술명으로 표시:

| 별칭 | 기술명 | 소스 체크포인트 | 설명 |
|------|--------|----------------|------|
| Balanced | Regression | `runs/ssl_finetune/best.pt` | Huber + V-shaped β, 안정적 예측 |
| Precise | Classification | `runs_ablation/cls_head/best.pt` | 128-bin CE + expectation decode, 가장 낮은 MAE |
| Creative | Stochastic | `runs_ablation/stoch_head/best.pt` | Gaussian NLL, temperature로 다양성 조절 |

- 모델별 개별 ONNX export (3개 파일)
- ControlPredictorMLP도 ONNX로 변환 (`control_mlp.onnx`)
- 3개 모델 모두 `enable_controls=true`, `time_scale=1.0`, **동일한 stats.json**
- Control MLP도 동일 stats로 학습되어 호환
- **모델 로딩**: 첫 선택 시 로드 + 캐시 (이후 즉시 전환)

### ONNX Export 세부사항

- **Batch**: Fixed batch=1 (window 하나씩 처리)
- **seq_len**: 고정 256
- **EMA weights** 사용 (학습 시 best checkpoint의 shadow weights)

## Control UI

### 레이블링

직관적 자연어 레이블 사용:
- **다이나믹 범위** (expressiveness/std): 좁음 ↔ 넓음
- **전체 세기** (dynamics_center/mean): 약하게 ↔ 강하게

### 동작 방식: Global + MLP Auto

1. 기본 모드: ControlPredictorMLP가 window별로 자동 예측
2. 유저가 슬라이더 조작 시: **절대값 조절** 방식
   - 슬라이더 중앙 = MLP 예측값 그대로 사용
   - 좌우 이동 시 [0, 1] 범위 내에서 절대값으로 변경
   - 모든 window에 동일한 값 적용 (global override)

### 재추론 UX

- **수동 버튼** ("Generate") 클릭 시 재추론
- 슬라이더 변경만으로는 자동 재추론하지 않음 (CPU 서버 부하 방지)

## Stochastic 모델 전용 UI

- **Temperature 슬라이더**: 0.0 ~ 2.0 (기본값 1.0)
  - 0.0: deterministic (mean과 동일)
  - 1.0: 학습된 분산 그대로
  - 2.0: 높은 다양성 (창의적 결과)
- Sampling 방식: `velocity = mu + temperature * sigma * epsilon` (epsilon ~ N(0,1))
- Regression/Classification 선택 시 temperature 슬라이더 숨김

## 시각화: Custom JS Piano Roll

### 기술 스택

- **Gradio CustomComponent** (JavaScript)
- **Canvas API** 사용 (수천 개 노트에서도 고성능)

### 기능

1. **Piano Roll 영역**
   - 가로축: 시간(sec), 세로축: pitch (MIDI note number)
   - 노트를 사각형으로 표시 (onset~offset × pitch)
   - Velocity → 색상 그라디언트 (FL Studio 스타일: 초록→빨강)
   - 줌/스크롤 인터랙션
   - Hover 시 노트 정보 (pitch, velocity, onset) 툴팁

2. **하단 Velocity Bar 영역**
   - FL Studio 스타일 velocity 막대 그래프
   - 각 노트에 대응하는 세로 막대
   - Piano roll과 가로축 동기화

### 색상 스키마

- Velocity 0 → 초록 (#00FF00)
- Velocity 127 → 빨강 (#FF0000)
- 선형 보간 그라디언트

## MIDI 입출력

### 입력 (MIDI → NoteEvent)

- `pretty_midi`로 .mid 파일 파싱
- Demo 내부에 MIDI→NoteEvent 변환 로직 구현
- **악기 경고**: 피아노 MIDI에 최적화되어 있다는 안내 메시지 표시

### 출력 (예측 velocity 반영)

- **원본 MIDI 복사 + velocity만 교체** 방식
  - sustain pedal, tempo, time signature, program change 등 메타데이터 보존
  - pretty_midi로 원본 파싱 → note.velocity만 예측값으로 교체 → 저장
- A/B 비교 없이 예측값만 출력

### 입력 제한

- 길이/노트 수 제한 없음
- 긴 MIDI는 프로그레스 바로 처리 상태 표시

## Audio Preview

- **FluidSynth** (서버사이드 합성)
- SoundFont(.sf2) ~30MB를 레포에 포함
- HF Spaces `packages.txt`에 `fluidsynth` 추가
- MIDI → WAV 변환 후 Gradio Audio 컴포넌트로 재생

## 추론 파이프라인 (Demo 독립 모듈)

기존 eval_baseline.py를 재사용하지 않고 demo 전용으로 새로 작성:

```
1. MIDI 파싱 (pretty_midi)
   └→ NoteEvent 리스트

2. Feature 계산
   └→ duration, IOI, delta_pitch, density, chord_size, register

3. Normalization (stats.json)
   └→ z-score 정규화

4. Windowing (size=256, stride=128)
   └→ WindowRecord 리스트

5. Control 예측
   ├→ Auto: ControlPredictorMLP(ONNX) per window
   └→ Manual: 유저 global 값으로 전체 override

6. ONNX Inference (batch=1, 순차 처리)
   ├→ Regression: output → denormalize
   ├→ Classification: logits → softmax → expectation → denormalize
   └→ Stochastic: (mu, sigma) → sample with temperature → denormalize

7. Reconstruction (center priority)
   └→ 겹치는 window 중 중앙에 가까운 예측값 선택

8. 결과 생성
   ├→ 원본 MIDI velocity 교체
   ├→ Piano roll 데이터 (JS 컴포넌트용 JSON)
   └→ Audio 합성 (FluidSynth)
```

## 파일 구조 (예상)

```
v3/demo/
├── app.py                  # Gradio 메인
├── inference.py            # 추론 파이프라인 (MIDI→예측→결과)
├── midi_io.py              # MIDI 파싱/저장 (pretty_midi)
├── features.py             # Feature 계산 + normalization
├── windowing.py            # Windowing + reconstruction
├── export_onnx.py          # PyTorch → ONNX 변환 스크립트
├── components/
│   └── piano_roll/         # Gradio CustomComponent (JS)
│       ├── frontend/
│       │   ├── index.js    # Canvas 기반 piano roll + velocity bars
│       │   └── style.css
│       └── __init__.py     # Python backend
├── models/                 # ONNX 모델 (stats 통일)
│   ├── regression.onnx
│   ├── classification.onnx
│   ├── stochastic.onnx
│   ├── control_mlp.onnx
│   └── stats.json          # 3개 모델 공통 (동일 stats)
├── assets/
│   └── piano.sf2           # SoundFont (~30MB)
├── requirements.txt
└── packages.txt            # HF Spaces 시스템 패키지 (fluidsynth)
```

## 주의사항

### stats.json 통일 확인

Demo에 사용하는 3개 모델은 모두 `enable_controls=true`, `time_scale=1.0`으로 학습되어
**stats.json이 동일** (md5: `2d0ca7a...`). Control MLP도 같은 stats로 학습되어 호환됨.

참고: `full_reg_b3`/`full_cls_b3`는 `enable_controls=false`, `time_scale=0.01`로
다른 stats를 사용하지만, demo에서는 사용하지 않음.

### 의존성 (Demo 런타임)

```
onnxruntime
pretty_midi
numpy
gradio
```

- **PyTorch 불필요** (모든 모델이 ONNX)
- **scikit-learn 불필요** (RandomForest 미사용, MLP도 ONNX)
- FluidSynth는 시스템 패키지로 설치
