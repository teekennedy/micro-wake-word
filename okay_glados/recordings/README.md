# Hand-recorded "Okay GLaDOS" samples

Drop audio files of a real person saying **"Okay GLaDOS"** into this folder.
They are mixed into the training data alongside the synthetic Piper samples,
which is the most effective way to make the model recognise a specific voice
that the synthetic voices don't cover well.

## What to record

Two ways, and they can be mixed:

* **One utterance per file.** Say "Okay GLaDOS" once, then stop. Trim is
  automatic — a little silence at either end is fine.
* **A whole session in one file.** Hit record, say "Okay GLaDOS" over and over
  with a pause of a second or two between takes, move around the room, vary
  the delivery. Commands after the wake word ("Okay GLaDOS, turn on the
  kitchen lights") and chatter in between are fine. Any file longer than
  4 s is treated as a session and has to be split by
  `tools/segment_recordings.py` first (see below).
* Any format ffmpeg can read: `.wav`, `.flac`, `.m4a`, `.mp3`, `.ogg`, `.opus`,
  `.aac`, `.aiff`. Any sample rate; everything is converted to 16 kHz mono.
* Filenames don't matter. Subfolders are fine (e.g. one per speaker).

## How many, and how to vary them

Aim for **30–60 recordings per person**. Twenty is enough to help; a hundred is
better. Variety matters far more than count — the augmentation pipeline adds
noise, reverb and gain for you, but it cannot invent a new way of *saying* it.
Deliberately vary:

| Vary this | Examples |
|---|---|
| Distance | right at the mic, arm's length, across the room |
| Volume | quiet/muttered, normal, called out from another room |
| Speed | slow and clear, normal, rushed |
| Intonation | statement, question, annoyed, sing-song |
| Position | facing the mic, facing away, from the side |
| Room | kitchen, living room, hallway — different reverb |
| Background | silent, TV on, dishwasher running |

Record on the **device you'll actually use**, or one like it. A phone recording
from across the kitchen is more useful training data than a studio-quality
close-mic take.

## Optional: negative recordings

Anything you put in `hard_negatives/` is trained as a *negative* — audio that
must **not** trigger the wake word. Good candidates: your household saying
similar phrases ("okay", "okay Google", a name that rhymes), or an hour of the
room just being lived in — conversation, TV, the dishwasher. This is the fix if
the finished model false-triggers on something specific in your home.

Long room recordings are cut into 5-minute blocks: every third block is held
out and streamed in step 5 as a **home false-accept test** ("Home room audio"
in the report), the rest is cut into 3 s training chunks. The segmenter
transcribes the file and drops any chunk near something that sounds like the
wake word, but don't say it on purpose.

## After adding files

If any file is a session (longer than 4 s), split it first. This needs a
whisper.cpp model (`ggml-large-v3-turbo.bin` recommended — the OpenSuperWhisper
app keeps one in `~/Library/Application Support/ru.starmel.OpenSuperWhisper/whisper-models/`):

```sh
uv run --no-project --with pywhispercpp --with soundfile --with numpy \
    python okay_glados/tools/segment_recordings.py --model path/to/ggml-large-v3-turbo.bin
```

Each stretch of speech is transcribed on its own, with no prompt. "okay +
one word" becomes a wake-word clip — Whisper spells GLaDOS as "Gladys",
"Klaus", "Blados" and so on, and in a wake-word session that is still the wake
word. Commands are cut off after the wake word, and four-or-more-word speech
with nothing okay/GLaDOS-like in it becomes an extra negative. Short garbled
fragments ("you", "Thank you", "Kick gloves") are dropped from both classes:
they are usually a quiet take Whisper failed on, and training them as
negatives would teach the model to miss exactly those. Check
`okay_glados/work/recordings_segmented/manifest.tsv` to see every decision.

Then:

```sh
.venv/bin/python okay_glados/step3_build_features.py --only recorded
.venv/bin/python okay_glados/step4_train.py
.venv/bin/python okay_glados/step5_evaluate.py
```

The split is deterministic: with 10+ recordings, 60% go to training and 20%
each to the validation and test sets, so the reported false-reject rate
includes real recorded speech the model never trained on. With fewer than 10,
all of them go to training and a warning is printed.
