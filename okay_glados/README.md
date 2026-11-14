# "Okay GLaDOS" — a microWakeWord training pipeline

A complete, reproducible pipeline that trains an ESP32-ready wake word model for
**"Okay GLaDOS"** and benchmarks it honestly. Five scripts, run in order.

```sh
uv venv --python 3.10 .venv        && uv pip install -e . 'datasets<4' matplotlib \
    'git+https://github.com/whatsnowplaying/audio-metadata@d4ebb238e6a401bb1a5aaaac60c9e2b3cb30929f'
uv venv --python 3.10 .venv-piper  && uv pip install --python .venv-piper \
    torch torchaudio 'piper-phonemize-cross==1.2.1' webrtcvad-wheels 'numpy<2'

.venv/bin/python okay_glados/step1_download_data.py    # ~30 GB of datasets
.venv/bin/python okay_glados/step2_generate_tts.py     # 23k Piper samples
.venv/bin/python okay_glados/step3_build_features.py   # spectrograms
.venv/bin/python okay_glados/step4_train.py            # trains + converts to TFLite
.venv/bin/python okay_glados/tools/select_checkpoint.py --apply   # pick the best checkpoint
.venv/bin/python okay_glados/step4_train.py --train 0  # convert the chosen weights
.venv/bin/python okay_glados/step5_evaluate.py         # benchmark, plots, manifest
```

Two virtualenvs are needed because piper-sample-generator pins `numpy<2` while
microWakeWord requires `numpy>=2`. `config.py` holds every path, count and
hyperparameter; it is the only file you should normally need to edit.

## Result

Measured on the quantized streaming model against 16.15 h of held-out ambient
audio and 1,250 held-out utterances from speakers the model never trained on:

| | at the shipped cutoff 0.71 |
|---|---|
| false accepts | 1 event in 16.15 h (0.062/h) |
| false rejects, normal-use speech | 4.40% |
| false rejects, adverse conditions | 18.64% |
| near-miss phrases accepted | 2.67% |

Cutoff 0.55 trades up to ~3 false accepts per 16 h for 1.60% false rejects;
cutoff 0.84 gives none at all for 9.12%. The full curve is in
`report/REPORT.md` and `report/roc_curve.png`, and the cutoff is a one-line
change in `report/okay_glados.json`.

### Running outside the source tree

Every path comes from `config.py`. These environment variables move the
writable parts elsewhere, so the pipeline can run from a read-only install with
its data on volumes (the wake word trainer in `k8s-homelab` does this):

| Variable | Default | What |
|---|---|---|
| `OKAY_GLADOS_WORK_DIR` | `okay_glados/work` | datasets, TTS output, features, checkpoints |
| `OKAY_GLADOS_LOG_DIR` | `okay_glados/logs` | training logs |
| `OKAY_GLADOS_REPORT_DIR` | `report/` or `experiments_v2/report_<run>/` | report, plots, `metrics.json`, model |
| `OKAY_GLADOS_RECORDINGS_DIR` | `okay_glados/recordings` | hand recordings |
| `OKAY_GLADOS_DEVICE_CLIPS` | `<work>/device_clips.tsv` | device captures, already trimmed and split (see `config.py`) |
| `OKAY_GLADOS_TRAIN_PYTHON`, `OKAY_GLADOS_PIPER_PYTHON` | `.venv`, `.venv-piper` | the two interpreters |

`step5_evaluate.py` always writes `metrics.json` next to the report. With
`--model <tflite> --cutoff <c>` it benchmarks another model (the published one)
on the same held-out data at that model's own cutoff, writing only
`metrics.json`.

---

## The pronunciation problem (read this before regenerating samples)

Piper phonemises text with eSpeak, and eSpeak reads `GLaDOS` as an acronym.
Verified with `piper_phonemize.phonemize_espeak` and confirmed by transcribing
the generated audio back with Whisper:

| text given to Piper | eSpeak phonemes | verdict |
|---|---|---|
| `Okay GLaDOS` | `oʊkˈeɪ dʒˈiː lˌæ dˈɑːs` | read as an acronym — "okay gee-la-doss" ❌ |
| `Okay glados` | `oʊkˈeɪ ɡl**ˈɑːd**oʊz` | "okay GLAH-doze", wrong vowel ❌ |
| **`Okay gladohs`** | **`oʊkˈeɪ ɡlˈædoʊz`** | **"okay GLAD-ohz" ✅** |

So generation uses the phoneticised spelling **`Okay gladohs`**, never the
literal name. Three further spellings in `config.POSITIVE_SPELLINGS` add
variety — a comma pause, a voiceless final /s/ (`gladohce`), weaker stress on
"o-" (`Ok gladohs`) — and every one of them keeps the `ɡlˈæd` nucleus.

That last point is a trap worth knowing about: `Okay glahdohs` *looks* like a
harmless variant but phonemises to `ɡlˈɑːdoʊz`, byte-for-byte the same as the
rejected `Okay glados`. Whisper can't hear the difference reliably either — it
transcribes both as "Okay, GLaDOS". So `check_pronunciation.py` asserts on
phonemes, not on transcripts: every configured spelling must contain
`config.REQUIRED_PHONEMES` (`ɡlˈæd`) and none may contain
`config.FORBIDDEN_PHONEMES` (`ɡlˈɑːd`). Re-run it after any edit to that list:

```sh
.venv-piper/bin/python okay_glados/tools/check_pronunciation.py
```

---

## Adding hand-recorded samples

Drop audio of a real person saying "Okay GLaDOS" into **`okay_glados/recordings/`**,
then:

```sh
# only if any file is a multi-take session or long room recording:
uv run --no-project --with pywhispercpp --with soundfile --with numpy \
    python okay_glados/tools/segment_recordings.py --model ggml-large-v3-turbo.bin
.venv/bin/python okay_glados/step3_build_features.py --only recorded
.venv/bin/python okay_glados/step4_train.py
.venv/bin/python okay_glados/step5_evaluate.py
```

See `recordings/README.md` for what and how much to record, and how sessions
are split. Held-out blocks of any room audio in `hard_negatives/` become an
extra "Home room audio" false-accept test in step 5, and checkpoint selection
weighs recall on the held-out recorded voices equally with the synthetic set. Recordings are
converted to 16 kHz mono automatically, split 60/20/20 into train/validation/test
(deterministically), and rendered many times with different augmentation so a
few dozen clips carry real weight. Audio placed in `recordings/hard_negatives/`
is trained as a negative instead — that is the lever to pull if the finished
model false-triggers on something specific in your home.

---

## How the pipeline works

**Step 1 — data.** Room impulse responses (MIT), background noise (AudioSet
balanced-train) and music (FMA) for augmentation; a *disjoint* AudioSet eval
slice reserved as extra ambient test audio; and the pre-computed negative
spectrogram sets from `kahrendt/microwakeword` (CHiME-6, VOiCES, FMA-medium,
FSD50K, WHAM!, DiPCo — roughly 900 hours).

Those negative sets are **10 ms window-step** features. The dataset card says
20 ms, but that describes the older `*_background` archives: the CHiME-6
training mmap holds 40.49 h of features when read at 10 ms, exactly matching the
corpus's published 40 h duration (and DiPCo 5.33 h, CHiME-6 dev+eval 9.67 h).
This matters — mixing 20 ms negatives with 10 ms positives would let the model
cheat on frame-rate artefacts and quietly inflate every benchmark.

**Step 2 — sample generation.** 12,000 wake-word samples for training and 2,000
for validation/testing, plus 8,000 near-miss "adversarial" phrases ("okay
glasses", "gladiolus", "okay Google", "glad hose", …) that must *not* fire.
Speakers 0–699 of LibriTTS-R generate training audio; speakers 700–899 generate
the held-out sets, so the reported false-reject rate is measured on voices the
model never heard. (`generate_samples.py` was patched with a `--min-speaker`
flag to make that split possible.) 30% of samples are generated inside a longer
carrier phrase so their prosody is that of connected speech.

**Step 3 — features.** Each clip is augmented (background noise at −5…+12 dB
SNR, room reverb, EQ, pitch, gain, distortion) into a 3.2 s window with the wake
word ending ~200 ms before the end, then converted to 40-channel microfrontend
spectrograms and stored as mmap-ninja arrays. Ten one-frame-shifted copies are
kept per render, which is what makes non-streaming training match streaming
inference. Work is sharded across 8 processes.

**Step 4 — training.** A MixConv streaming network (32-filter strided input
convolution, four mixed-depthwise blocks, one inference every 30 ms) — the same
topology as the production ESPHome v2 models. Best weights are selected by
*average viable recall*: the mean recall achievable anywhere between 0 and 2
false accepts per hour. Training then converts and quantizes to a streaming
TFLite model.

**Between 4 and 5 — checkpoint selection.** microWakeWord keeps the weights that
maximise `average_viable_recall` on the non-streaming validation pass. That
metric turned out to be a poor proxy for the artefact that actually ships: over
one run, checkpoints scoring *higher* on it streamed at 12% false rejects while
a lower-scoring one streamed at 5%. `tools/select_checkpoint.py` therefore
converts candidate checkpoints exactly as the release build does and measures
each as a quantized streaming model on the **validation** ambient audio, keeping
the genuine best. The test sets stay untouched for step 5.

**Step 5 — evaluation.** The quantized streaming model is run end to end over
every hour of held-out ambient audio with ESPHome's own 5-frame moving average
and detection cooldown, and over every held-out positive clip. That produces the
false-accept/false-reject curve, the operating cutoff, `training_progress.png`,
`roc_curve.png` and the ESPHome manifest.

---

## Using the model

`report/okay_glados.json` and `report/okay_glados.tflite` are ready for ESPHome:

```yaml
micro_wake_word:
  models:
    - model: <path or URL to okay_glados.json>
```

`probability_cutoff` in that manifest is the measured operating point. Raise it
if you get false triggers in your home; lower it if the wake word is missed.
Every 0.01 step is a point on `roc_curve.png`, so you can see exactly what the
trade costs.

## What didn't work

Recorded so the next person doesn't repeat them:

* **Removing SpecAugment** made it clearly worse (15.11% vs 6.76% mean false
  rejects), despite the earlier run suggesting it was hurting.
* **Rebalancing the class weights** away from microWakeWord's defaults
  (negative class weight 20 → 10, positives and adversarials up) was the worst
  run of the four at 19.37%. The defaults are well chosen.
* **Training longer** does not help. Every run peaked between steps 10k and 17k
  and drifted afterwards. Run-to-run variance is larger than the effect of most
  hyperparameters, which is exactly why the checkpoint sweep matters.
