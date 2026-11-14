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
| 0.50 | 5.028 | 0.72% | 4.08% | 14.29% | 10.17% |
| 0.55 | 4.422 | 0.88% | 4.88% | 14.29% | 9.00% |
| 0.60 | 3.816 | 1.04% | 5.84% | 14.29% | 8.17% |
| 0.65 | 3.089 | 1.36% | 6.40% | 14.29% | 7.33% |
| 0.70 | 2.665 | 2.08% | 7.76% | 14.29% | 7.00% |
| 0.75 | 2.302 | 3.12% | 9.12% | 14.29% | 6.00% |
| 0.80 | 1.878 | 4.56% | 11.84% | 14.29% | 5.33% |
| 0.85 | 1.454 | 6.16% | 15.92% | 14.29% | 4.00% |
| 0.90 | 0.788 | 8.00% | 19.44% | 14.29% | 3.50% |
| 0.95 | 0.363 | 10.64% | 22.56% | 16.07% | 2.83% |
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
- the framework's own metric (average viable recall) peaks at 94.19% on step 18000
- but the shipped weights were chosen by converting candidate checkpoints to quantized streaming models and measuring them (`tools/select_checkpoint.py`); see `checkpoint_selection.json`
- see `training_progress.png`
