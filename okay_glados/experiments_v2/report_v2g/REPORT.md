# "Okay GLaDOS" — microWakeWord training report

All numbers below come from the **quantized streaming TFLite model** (the artefact that runs on the ESP32), measured on data that was never trained on.

## Headline result

| metric | value | target | |
|---|---|---|---|
| probability cutoff | 0.87 | — | |
| false accepts | 0.061/h (1 event in 16.5 h) | ≤ 0.1/h | ✅ |
| false reject rate (normal use) | 5.87% | ≤ 5% | ❌ |

The model **misses** the false-reject target at the most sensitive cutoff that holds false accepts to 0.1 per hour or below.

Raising the cutoff to **0.90** gives *no* false accept anywhere in 16.51 hours, at 6.68% false rejects. That is the setting to use if a spurious wake is more annoying than a missed one. Note that zero observed events over a finite test set is not a true zero: by the rule of three the 95% upper bound is still **0.18 per hour**.

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
| 0.50 | 1.272 | 0.64% | 6.72% | 18.75% | 7.17% |
| 0.55 | 1.151 | 0.72% | 7.76% | 19.64% | 7.00% |
| 0.60 | 0.788 | 1.04% | 8.64% | 20.54% | 6.33% |
| 0.65 | 0.666 | 1.44% | 9.76% | 20.54% | 5.50% |
| 0.70 | 0.485 | 1.60% | 10.72% | 20.54% | 5.17% |
| 0.75 | 0.424 | 2.08% | 12.16% | 21.43% | 5.00% |
| 0.80 | 0.303 | 2.88% | 13.84% | 22.32% | 4.33% |
| 0.85 | 0.182 | 3.76% | 17.12% | 23.21% | 3.67% |
| 0.87 | 0.061 | 4.32% | 17.60% | 23.21% | 3.50% |
| 0.90 | 0.000 | 5.20% | 19.44% | 23.21% | 3.17% |
| 0.95 | 0.000 | 7.36% | 23.04% | 24.11% | 2.50% |
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
- the framework's own metric (average viable recall) peaks at 95.21% on step 18000
- but the shipped weights were chosen by converting candidate checkpoints to quantized streaming models and measuring them (`tools/select_checkpoint.py`); see `checkpoint_selection.json`
- see `training_progress.png`
