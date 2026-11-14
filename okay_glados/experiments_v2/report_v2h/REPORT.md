# "Okay GLaDOS" — microWakeWord training report

All numbers below come from the **quantized streaming TFLite model** (the artefact that runs on the ESP32), measured on data that was never trained on.

## Headline result

| metric | value | target | |
|---|---|---|---|
| probability cutoff | 1.00 | — | |
| false accepts | **0 observed** in 16.5 h | ≤ 0.1/h | ✅ |
| false reject rate (normal use) | 100.00% | ≤ 5% | ❌ |

The model **misses** the false-reject target at the most sensitive cutoff that holds false accepts to 0.1 per hour or below.

Raising the cutoff to **1.00** gives *no* false accept anywhere in 16.51 hours, at 100.00% false rejects. That is the setting to use if a spurious wake is more annoying than a missed one. Note that zero observed events over a finite test set is not a true zero: by the rule of three the 95% upper bound is still **0.18 per hour**.

## Test data

| set | size |
|---|---|
| DiPCo dinner party (ambient, false accepts) | 5.33 h |
| AudioSet (held out) (ambient, false accepts) | 10.82 h |
| Home room audio (held out) (ambient, false accepts) | 0.36 h |
| Normal use (unseen speakers) (positives, false rejects) | 1250 clips |
| Adverse (noise/reverb/quiet) (positives, false rejects) | 1250 clips |
| Hand recorded (held out) (positives, false rejects) | 112 clips |
| Near-miss phrases (adversarial) | see table |

Total ambient audio: **16.51 hours**.

## Operating curve

| cutoff | false accepts/h | FRR Normal use (unseen speakers) | FRR Adverse (noise/reverb/quiet) | FRR Hand recorded (held out) | near-miss accepts |
|---|---|---|---|---|---|
| 0.50 | 16.174 | 0.16% | 2.32% | 14.29% | 14.67% |
| 0.55 | 13.630 | 0.16% | 3.04% | 14.29% | 12.67% |
| 0.60 | 11.086 | 0.32% | 3.52% | 14.29% | 11.67% |
| 0.65 | 8.905 | 0.72% | 4.48% | 14.29% | 10.33% |
| 0.70 | 6.845 | 1.04% | 5.44% | 14.29% | 9.50% |
| 0.75 | 5.391 | 1.44% | 6.32% | 14.29% | 8.83% |
| 0.80 | 3.998 | 1.84% | 7.44% | 14.29% | 7.67% |
| 0.85 | 3.150 | 2.48% | 10.00% | 14.29% | 5.83% |
| 0.90 | 2.060 | 3.76% | 12.56% | 14.29% | 5.17% |
| 0.95 | 1.090 | 5.92% | 16.40% | 14.29% | 3.50% |
| 1.00 | 0.000 | 100.00% | 100.00% | 100.00% | 0.00% |

## Model

- file: `okay_glados.tflite` (60.7 KiB)
- feature step size: 10 ms, stride 3 (one inference every 30 ms)
- ESPHome manifest: `okay_glados.json`

```yaml
micro_wake_word:
  models:
    - model: <url or path to okay_glados.json>
```

## Training

- 60 validation checkpoints over 30000 steps
- the framework's own metric (average viable recall) peaks at 92.72% on step 9000
- but the shipped weights were chosen by converting candidate checkpoints to quantized streaming models and measuring them (`tools/select_checkpoint.py`); see `checkpoint_selection.json`
- see `training_progress.png`
