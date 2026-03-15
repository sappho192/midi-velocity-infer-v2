# Demo Implementation Report

## 개요

`v3/demo/`에 Gradio 기반 MIDI Velocity Inference 데모를 구현했다.
유저가 MIDI 파일을 업로드하면 Transformer 모델이 노트별 velocity를 예측하고,
Plotly piano roll로 시각화 + FluidSynth로 오디오 합성 + MIDI 다운로드를 제공한다.

런타임에 PyTorch 의존성 없이 OnnxRuntime만으로 추론한다.

## 아키텍처

```
MIDI 업로드 (.mid)
    │
    ▼
pretty_midi 파싱 ──→ NoteEvent 리스트
    │
    ▼
Feature 계산 (duration, IOI, density, chord_size, register)
    │   TIME_SCALE=100 (real sec → centisecond 단위로 변환)
    ▼
z-score 정규화 (stats.json)
    │
    ▼
Windowing (size=256, stride=128)
    │
    ├──→ Control MLP (ONNX) ──→ per-window [expr, dyn] 예측
    │    또는 유저 manual override
    ▼
Velocity Model (ONNX, batch=1)
    │
    ├── Regression: output [1,256] → denormalize
    ├── Classification: logits [1,256,128] → softmax → expectation
    └── Stochastic: (mu, log_sigma) → sample with temperature
    │
    ▼
Center Priority Reconstruction
    │
    ▼
┌──────────────────────────────────────┐
│  Plotly Piano Roll (velocity 색상)   │
│  FLAC 오디오 합성 (FluidSynth)       │
│  MIDI 다운로드 (원본 메타데이터 보존)  │
└──────────────────────────────────────┘
```

## 사용 모델

| 별칭 | 기술명 | 소스 체크포인트 | ONNX 크기 |
|------|--------|----------------|-----------|
| Balanced | Regression | `runs/ssl_finetune/best.pt` | 15 MB |
| Precise | Classification | `runs_ablation/cls_head/best.pt` | 15 MB |
| Creative | Stochastic | `runs_ablation/stoch_head/best.pt` | 15 MB |
| — | Control MLP | `runs/ssl_finetune/control_mlp/control_mlp.pt` | 28 KB |

- 3개 모델 모두 `enable_controls=true`, `time_scale=1.0`, 동일 `stats.json` (md5: `2d0ca7a`)
- EMA shadow weights 사용
- ONNX opset 14 (legacy exporter, MultiheadAttention 호환)

## ONNX Export

`export_onnx.py`로 일괄 변환:

```bash
uv run python demo/export_onnx.py \
  --reg-ckpt runs/ssl_finetune/best.pt \
  --cls-ckpt runs_ablation/cls_head/best.pt \
  --stoch-ckpt runs_ablation/stoch_head/best.pt \
  --mlp-ckpt runs/ssl_finetune/control_mlp/control_mlp.pt \
  --stats runs/ssl_finetune/stats.json \
  --output-dir demo/models
```

### Export 시 해결한 문제

- **torch.export 기반 새 exporter 실패**: `nn.MultiheadAttention`의 view reshape에서 shape 오류 발생. `dynamo=False`로 legacy TorchScript exporter 사용하여 해결.
- **opset 17 → 14**: onnxscript 버전 변환 경고 방지를 위해 opset 14 사용.

## TIME_SCALE 문제 발견 및 해결

### 문제

MAESTRO CSV 데이터의 `time` 컬럼은 centisecond 단위 (예: `784` = 7.84초).
학습 시 `time_scale=1.0`으로 이 값을 그대로 사용했으므로, 모델이 학습한
feature 스케일은 "centisecond as seconds" 단위이다.

반면 demo에서 `pretty_midi`는 실제 초 단위를 반환하므로, 그대로 사용하면
duration/IOI가 ~100배 작아져서 z-score 정규화 후 feature가 왜곡된다.

### 초기 증상

```
Regression (auto): velocity range 18-30, mean 18.6  ← 비정상
```

### 해결

`features.py`에 `TIME_SCALE = 100.0`을 적용하여 real seconds를
centisecond 단위로 변환:

```python
scaled_onsets = [e.onset_sec * TIME_SCALE for e in events]
event.duration_sec = max(scaled_offsets[idx] - scaled_onsets[idx], 0.0)
```

### 수정 후 결과

```
Regression (auto):           range 27-118, mean 93.1, std 10.5
Classification (auto):       range 38-114, mean 90.1, std 8.6
Stochastic (auto, temp=1):   range 1-127,  mean 83.6, std 21.2
Regression (expr=0.8,dyn=0.7): range 5-113, mean 79.8, std 16.8
Regression (expr=0.2,dyn=0.3): range 7-88,  mean 54.9, std 13.5
```

Control 파라미터에 따른 velocity 변화도 정상 확인.

## Piano Roll 시각화

### 초기 구현: Canvas + `gr.HTML`

Gradio 6.9.0이 `gr.HTML` 내 `<script>` 태그를 sanitize하여
Canvas 기반 piano roll이 렌더링되지 않는 문제 발생.

### 최종 구현: Plotly + `gr.Plot`

- `go.Bar(orientation="h")`로 노트 사각형 표현 (색상: velocity 기반 초록→빨강)
- 하단 `go.Bar`로 velocity 막대 그래프
- `make_subplots`로 piano roll(70%) + velocity bar(30%) 구성
- `shared_xaxes=True`로 가로축 동기화

### 성능 최적화

| 방식 | 6,290 노트 렌더링 | JSON 크기 |
|------|-------------------|-----------|
| `add_shape` (개별) | timeout (수 분) | — |
| `go.Bar` horizontal (최종) | **0.19초** | **762 KB** |

## 오디오 합성

- **엔진**: FluidSynth CLI → 임시 WAV → FLAC L5 변환
- **SoundFont**: Nice-Steinway-Lite-v3.0.sf2 (71 MB)
- **출력 포맷**: FLAC (compression level 5)
  - WAV 109 MB → FLAC 28 MB (약 3.9배 압축)
- **Fallback**: FluidSynth CLI 없으면 `pretty_midi.fluidsynth()` 사용

## MIDI 출력

- `pretty_midi`로 원본 MIDI deep copy → `note.velocity`만 예측값으로 교체
- Sustain pedal, tempo, time signature, program change 등 메타데이터 보존
- Velocity 범위: [1, 127] clamping

## 파일 구조

```
v3/demo/
├── app.py              # Gradio 앱 (좌우 분할 UI)
├── inference.py        # VelocityInferenceEngine (ONNX 파이프라인)
├── midi_io.py          # MIDI 파싱/출력 (pretty_midi)
├── features.py         # Feature 계산 + z-score 정규화
├── windowing.py        # Windowing + 21-dim 특징 추출 + 복원
├── piano_roll.py       # Plotly 기반 piano roll 시각화
├── audio.py            # FluidSynth FLAC 합성
├── export_onnx.py      # PyTorch→ONNX 변환 (dev-time only)
├── requirements.txt    # Runtime 의존성
├── packages.txt        # HF Spaces 시스템 패키지
├── models/
│   ├── regression.onnx     # 15 MB
│   ├── classification.onnx # 15 MB
│   ├── stochastic.onnx     # 15 MB
│   ├── control_mlp.onnx    # 28 KB
│   └── stats.json          # 공통 normalization 통계
└── assets/
    └── Nice-Steinway-Lite-v3.0.sf2  # 71 MB
```

## 런타임 의존성

```
onnxruntime    # ONNX 추론
gradio         # Web UI
pretty_midi    # MIDI 파싱/출력
numpy          # 수치 연산
soundfile      # FLAC 인코딩
plotly         # Piano roll 시각화
fluidsynth     # 시스템 패키지 (오디오 합성)
```

PyTorch, scikit-learn 불필요.

## 실행 방법

```bash
cd v3/demo
uv run python app.py
```

## 배포 (HuggingFace Spaces)

`v3/demo/` 내에서 개발하고, HF Space 배포 시 필요 파일만 추출:
- `*.py` (export_onnx.py 제외)
- `models/` 디렉토리 전체
- `assets/` 디렉토리 전체
- `requirements.txt`, `packages.txt`
