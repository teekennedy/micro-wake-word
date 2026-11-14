# "Okay GLaDOS" — microWakeWord training report

All numbers below come from the **quantized streaming TFLite model** (the artefact that runs on the ESP32), measured on data that was never trained on.

## Headline result

| metric | value | target | |
|---|---|---|---|
| probability cutoff | 0.71 | — | |
| false accepts | 0.061/h (1 event in 16.5 h) | ≤ 0.1/h | ✅ |
| false reject rate (normal use) | 8.08% | ≤ 5% | ❌ |

The model **misses** the false-reject target at the most sensitive cutoff that holds false accepts to 0.1 per hour or below.

Raising the cutoff to **0.83** gives *no* false accept anywhere in 16.51 hours, at 12.56% false rejects. That is the setting to use if a spurious wake is more annoying than a missed one. Note that zero observed events over a finite test set is not a true zero: by the rule of three the 95% upper bound is still **0.18 per hour**.

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
| 0.50 | 0.363 | 1.36% | 11.44% | 34.82% | 3.67% |
| 0.55 | 0.182 | 1.60% | 12.56% | 38.39% | 3.50% |
| 0.60 | 0.121 | 2.32% | 14.24% | 41.07% | 3.33% |
| 0.65 | 0.121 | 3.04% | 16.24% | 44.64% | 3.00% |
| 0.70 | 0.121 | 3.92% | 18.32% | 47.32% | 2.67% |
| 0.71 | 0.061 | 4.40% | 18.64% | 49.11% | 2.67% |
| 0.75 | 0.061 | 5.20% | 21.04% | 50.00% | 2.50% |
| 0.80 | 0.061 | 7.36% | 24.80% | 53.57% | 2.17% |
| 0.85 | 0.000 | 9.92% | 28.56% | 58.93% | 1.83% |
| 0.90 | 0.000 | 13.28% | 33.20% | 66.07% | 1.50% |
| 0.95 | 0.000 | 19.12% | 40.32% | 70.54% | 0.83% |
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
- the framework's own metric (average viable recall) peaks at 95.97% on step 10500
- but the shipped weights were chosen by converting candidate checkpoints to quantized streaming models and measuring them (`tools/select_checkpoint.py`); see `checkpoint_selection.json`
- see `training_progress.png`

### Training runs tried

Each run was swept checkpoint by checkpoint as a quantized streaming model against the *validation* ambient audio; the best of the best was shipped. Lower is better.

| run | what changed | candidates | best step | mean FRR (0–2 accepts/h) |
|---|---|---|---|---|
| B | batch 256, 18k @1e-3 then 12k @3e-4  **(shipped)** | 24 | 10000 | 6.76% |
| C | SpecAugment removed, 3 LR stages | 30 | 9500 | 15.11% |
| D | single 18k @1e-3 stage, dense checkpoints | 36 | 16000 | 8.67% |
| E | rebalanced weights (neg 10, pos 3, adversarial 6) | 36 | 17000 | 19.37% |

The two hypotheses that failed are worth recording: removing SpecAugment (run C) and rebalancing the class weights away from microWakeWord's defaults (run E) both made things clearly worse.
