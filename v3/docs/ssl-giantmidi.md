# SSL Pretraining on GiantMIDI-Piano → Fine-tune on MAESTRO

## Context

v3 모델이 MAESTRO 962곡만으로 훈련하여 심각한 과적합 (val MAE 9.91, test MAE 23.43). GiantMIDI-Piano (10,855곡, AI 전사)을 활용한 SSL pretraining으로 backbone representation을 먼저 학습한 후 MAESTRO로 fine-tune하여 일반화 성능 개선 목표.

**GiantMIDI-Piano**: AI 전사 velocity (ground truth 아님) → supervised 부적합, SSL에는 무관 (velocity는 SSL 입력 아님)

**현재 상태**: Phase 5a oracle-conditioned 훈련 진행 중. SSL 파이프라인은 독립적으로 구축 가능.

---

## Masked Note Modeling (MNM) 설계

### Pretext Task
Window 내 15%의 note를 마스킹하고, context로부터 원래 속성 예측:
- **Pitch prediction**: 128-class CE loss (마스킹된 note의 원래 pitch 예측)
- **Continuous feature reconstruction**: 6-dim MSE loss (duration, IOI, delta_pitch 등)

### Masking 방식
마스킹된 note의 전체 embedding output을 **learnable mask vector** `[d_model]`로 교체:
- pitch, register, continuous 모두 한번에 마스킹 → register에서 pitch 유추 방지
- Embedding(128/4) 크기 변경 불필요 → 기존 체크포인트 호환성 유지
- 구현이 단순: `x[mask_indices] = self.mask_embedding`

### 왜 velocity를 SSL 입력에서 제외하는가
- Velocity는 현재도 입력이 아닌 타겟 (continuous features에 미포함)
- GiantMIDI velocity는 AI 전사라 noisy → 입력에 넣으면 backbone이 noisy signal에 의존
- 변경 사항 없음: 기존 feature set (duration, IOI, delta_pitch, density, chord_size) 그대로 사용

---

## 구현 단계

### Step 1: GiantMIDI MIDI→CSV 변환

**새 파일**: `scripts/convert_giantmidi_to_csv.py`

- `pretty_midi` 사용하여 .mid → canonical CSV (piece_id, note_index, pitch, onset_sec, offset_sec, velocity)
- 메타데이터 TSV 파싱: `giant_midi_piano==1` 필터, `split` 컬럼으로 train/val/test 분리
- `audio_name` 컬럼 → MIDI 파일명 매칭 (`.mid` 확장자 추가)
- 병렬 처리 (ProcessPoolExecutor), 실패 파일 로깅 후 skip
- 출력: `{output_dir}/{train,validation,test}/*.csv`

**의존성**: `pyproject.toml`에 `pretty-midi` 추가 (optional)

| Split | 곡 수 |
|-------|------|
| train | 6,035 |
| validation | 3,224 |
| test | 1,596 |

### Step 2: PretrainModel 구현

**새 파일**: `mvi_v3/models/pretrain_model.py`

```
PretrainModel
├── NoteEmbedding (기존 그대로 재사용)
├── mask_embedding: nn.Parameter([d_model])  ← learnable MASK vector
├── T5RelativePositionBias (기존 재사용)
├── EncoderBlocks × num_layers (기존 재사용)
├── OutputNorm (기존 재사용)
├── pitch_head: Linear(d_model → d_model → 128)  ← pitch CE
└── continuous_head: Linear(d_model → d_model → 6)  ← feature MSE
```

- backbone 구조가 `TransformerVelocityModel`과 동일 (embedding, position_bias, layers, output_norm)
- `backbone_state_dict()` 메서드: 위 4개 모듈의 state_dict만 추출
- `mask_embedding`은 backbone에 포함하지 않음 (pretrain 전용)

**핵심**: backbone key 이름이 `TransformerVelocityModel`과 일치 → `load_state_dict(strict=False)`로 직접 전이

### Step 3: Pretrain Dataset

**새 파일**: `mvi_v3/data/pretrain_dataset.py`

기존 `WindowDataset` 패턴 재사용, MNM 마스킹 추가:

```python
__getitem__ returns:
  pitch, register_bucket, continuous, padding_mask  # 마스킹 전 원본 유지
  original_pitch  # CE target
  original_continuous  # MSE target
  mnm_mask  # [seq_len] bool, True=masked
```

마스킹은 모델 내부에서 적용 (mask_embedding 교체). Dataset은 mask 위치만 결정.

### Step 4: Pretrain Engine

**새 파일**: `mvi_v3/training/pretrain_engine.py`

`run_pretrain_epoch()` — `run_epoch()` 패턴 따름:
- Forward: `pitch_logits, cont_pred = model(batch, mnm_mask)`
- Pitch loss: `F.cross_entropy(pitch_logits[mnm_mask], original_pitch[mnm_mask])`
- Continuous loss: `F.mse_loss(cont_pred[mnm_mask], original_continuous[mnm_mask])`
- Total: `pitch_weight * pitch_loss + continuous_weight * cont_loss`
- 기존 인프라 재사용: gradient accumulation, grad clipping, EMA, scheduler

### Step 5: Pretrain CLI

**새 파일**: `mvi_v3/cli/pretrain_ssl.py`

`train_baseline.py` 구조 미러링:
1. GiantMIDI CSV 로딩 (`load_piece_directory` + `prepare_pieces` 재사용)
2. `fit_dataset_stats()` on GiantMIDI train → `pretrain_stats.json`
3. `build_windows()` → `PretrainWindowDataset`
4. `PretrainModel` 생성, AdamW, cosine scheduler
5. 훈련 루프 (val loss = MNM loss on GiantMIDI val)
6. 완료 시 `backbone.pt` 저장:
   ```python
   {"backbone_state_dict": model.backbone_state_dict(), "config": {...}, "epoch": N}
   ```

CLI args: `--train-dir, --val-dir, --output-dir, --epochs, --mask-ratio, --batch-size, --learning-rate`

### Step 6: Fine-tune 통합

**수정 파일**: `mvi_v3/cli/train_baseline.py`

`--pretrained-backbone` 인자 추가. 모델 생성 직후:
```python
if args.pretrained_backbone:
    ckpt = torch.load(args.pretrained_backbone, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["backbone_state_dict"], strict=False)
    print(f"Loaded backbone: missing={len(missing)} (head weights), unexpected={len(unexpected)}")
```

missing = velocity head weights (정상), unexpected = 없어야 함

---

## 파일 변경 요약

| 파일 | 액션 | 설명 |
|------|------|------|
| `scripts/convert_giantmidi_to_csv.py` | NEW | MIDI→CSV 변환기 |
| `mvi_v3/models/pretrain_model.py` | NEW | PretrainModel (backbone + MNM heads) |
| `mvi_v3/data/pretrain_dataset.py` | NEW | MNM 마스킹 dataset |
| `mvi_v3/training/pretrain_engine.py` | NEW | SSL 훈련 루프 |
| `mvi_v3/cli/pretrain_ssl.py` | NEW | SSL pretraining CLI |
| `mvi_v3/cli/train_baseline.py` | MODIFY | `--pretrained-backbone` 추가 |
| `pyproject.toml` | MODIFY | `pretty-midi` dep, `mvi-v3-pretrain` entry point |

기존 파일 변경 최소화 — `train_baseline.py`에 ~10줄, `pyproject.toml`에 2줄만 추가.

---

## 실행 예시

```bash
# 1. MIDI→CSV 변환 (1회)
uv run python scripts/convert_giantmidi_to_csv.py \
  --midi-dir /home/tikim/dataset/GiantMIDI-PIano/midis \
  --metadata /home/tikim/dataset/GiantMIDI-PIano/metadata/full_music_pieces_youtube_similarity_pianosoloprob_split.csv \
  --output-dir /home/tikim/dataset/GiantMIDI-PIano/csv

# 2. SSL Pretrain (~6K pieces, ~50 epochs)
uv run python -m mvi_v3.cli.pretrain_ssl \
  --train-dir /home/tikim/dataset/GiantMIDI-PIano/csv/train \
  --val-dir /home/tikim/dataset/GiantMIDI-PIano/csv/validation \
  --output-dir runs/ssl_pretrain \
  --epochs 50 --mask-ratio 0.15 --batch-size 256

# 3. Fine-tune on MAESTRO (pretrained backbone)
uv run python -m mvi_v3.cli.train_baseline \
  --train-dir /home/tikim/dataset/maestro/maestro-raw/maestro-midi/train \
  --val-dir /home/tikim/dataset/maestro/maestro-raw/maestro-midi/validation \
  --output-dir runs/ssl_finetune \
  --pretrained-backbone runs/ssl_pretrain/backbone.pt \
  --enable-controls --control-dims 2 \
  --augment-velocity-jitter 2 --embedding-dropout 0.1 --dropout 0.2 \
  --velocity-weight-beta 3.0

# 4. Eval & 비교
uv run python -m mvi_v3.cli.eval_baseline \
  --data-dir /home/tikim/dataset/maestro/maestro-raw/maestro-midi/test \
  --checkpoint runs/ssl_finetune/best.pt \
  --stats runs/ssl_finetune/stats.json \
  --output-dir runs/ssl_finetune/eval
```

## Verification

1. **Pretrain 수렴**: GiantMIDI val에서 MNM loss 감소 확인, pitch accuracy > 50% (random = 0.78%)
2. **Fine-tune 비교**: 동일 설정에서 pretrained vs scratch 비교
   - Test MAE 감소 여부 (현재 v3: 23.43)
   - Val-test gap 축소 여부 (과적합 완화)
3. **Controllability 유지**: Phase 5a control sweep 결과가 pretrained backbone에서도 작동하는지
