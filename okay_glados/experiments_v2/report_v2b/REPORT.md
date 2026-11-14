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
| 0.50 | 2.302 | 0.80% | 7.44% | 10.71% | 5.33% |
| 0.55 | 2.120 | 0.96% | 8.48% | 10.71% | 5.17% |
| 0.60 | 1.938 | 1.44% | 9.68% | 11.61% | 4.83% |
| 0.65 | 1.696 | 2.24% | 11.84% | 12.50% | 4.17% |
| 0.70 | 1.454 | 3.04% | 13.44% | 16.07% | 3.83% |
| 0.75 | 1.333 | 3.92% | 15.52% | 19.64% | 3.17% |
| 0.80 | 1.212 | 5.20% | 18.00% | 19.64% | 3.00% |
| 0.85 | 0.969 | 7.28% | 22.08% | 20.54% | 2.67% |
| 0.90 | 0.666 | 9.12% | 26.00% | 20.54% | 1.83% |
| 0.95 | 0.363 | 12.24% | 30.24% | 20.54% | 1.00% |
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
- the framework's own metric (average viable recall) peaks at 94.20% on step 13500
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
