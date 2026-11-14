#!/usr/bin/env python3
"""Step 2 -- generate the synthetic wake-word and adversarial samples with Piper.

Reminder about the spelling (see config.POSITIVE_SPELLINGS): Piper phonemises
via eSpeak, and the literal strings "Okay GLaDOS" and "Okay glados" are both
mispronounced.  Generation therefore uses the phoneticised "Okay gladohs" and a
handful of variants that phonemise to the same word.

Four sets come out of this step:

  positive_train/     wake word, LibriTTS-R speakers [0, 700)
  positive_heldout/   wake word, LibriTTS-R speakers [700, 900)   <- never trained on
  adversarial/train/  near-miss phrases, speakers [0, 700)
  adversarial/heldout near-miss phrases, speakers [700, 900)

Keeping the speaker ranges disjoint is what makes the reported false-reject
rate meaningful: the test voices are voices the model has never heard.
"""

import argparse
import random
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

GENERATOR = config.PIPER_DIR / "generate_samples.py"
MODEL = config.PIPER_DIR / "models" / "en_US-libritts_r-medium.pt"

# Speaking rates and vocal variability cycled across batches.  A wide spread
# here is the single cheapest way to make the model robust to how fast and how
# emphatically someone says the wake word.
LENGTH_SCALES = ["0.7", "0.85", "1.0", "1.15", "1.3", "1.45"]
NOISE_SCALES = ["0.5", "0.667", "0.8", "0.9", "1.0"]
NOISE_SCALE_WS = ["0.6", "0.8", "1.0"]
SLERP_WEIGHTS = ["0.0", "0.3", "0.5", "0.7", "1.0"]


def write_phrase_file(path: Path, weighted_phrases, n_lines=97, seed=1234):
    """Write a phrase list whose line frequencies match the requested shares.

    generate_samples.py cycles through the lines while it independently cycles
    through the (speaker, length, noise) settings, so a prime line count makes
    every phrase meet every setting combination.
    """
    lines = []
    for phrase, share in weighted_phrases:
        lines.extend([phrase] * max(1, round(share * n_lines)))
    random.Random(seed).shuffle(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def generate(phrase_file: Path, out_dir: Path, count: int, min_speaker: int,
             max_speaker: int, min_phoneme_count=None, batch_size=250):
    if out_dir.exists() and len(list(out_dir.glob("*.wav"))) >= count:
        print(f"  have {count} samples in {out_dir}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(config.PIPER_PYTHON), str(GENERATOR), str(phrase_file),
        "--model", str(MODEL),
        "--max-samples", str(count),
        "--batch-size", str(batch_size),
        "--min-speaker", str(min_speaker),
        "--max-speakers", str(max_speaker),
        "--output-dir", str(out_dir),
        "--length-scales", *LENGTH_SCALES,
        "--noise-scales", *NOISE_SCALES,
        "--noise-scale-ws", *NOISE_SCALE_WS,
        "--slerp-weights", *SLERP_WEIGHTS,
    ]
    if min_phoneme_count:
        cmd += ["--min-phoneme-count", str(min_phoneme_count)]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def count_wavs(d: Path):
    return len(list(d.glob("**/*.wav")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="regenerate from scratch")
    args = ap.parse_args()

    if args.force:
        for d in (config.POSITIVE_TRAIN_DIR, config.POSITIVE_HELDOUT_DIR, config.ADVERSARIAL_DIR):
            shutil.rmtree(d, ignore_errors=True)

    text_dir = config.WORK_DIR / "phrase_lists"
    positive_txt = write_phrase_file(text_dir / "positive.txt", config.POSITIVE_SPELLINGS)
    adversarial_txt = write_phrase_file(
        text_dir / "adversarial.txt",
        [(p, 1.0 / len(config.ADVERSARIAL_PHRASES)) for p in config.ADVERSARIAL_PHRASES],
        n_lines=len(config.ADVERSARIAL_PHRASES) * 3,
    )

    # --- positives ------------------------------------------------------- #
    # 70% plain utterances, 30% generated inside a longer carrier so the
    # prosody is that of connected speech rather than an isolated command.
    n_plain = int(config.N_POSITIVE_TRAIN * 0.7)
    n_carrier = config.N_POSITIVE_TRAIN - n_plain
    generate(positive_txt, config.POSITIVE_TRAIN_DIR / "plain", n_plain, 0,
             config.TRAIN_MAX_SPEAKER)
    generate(positive_txt, config.POSITIVE_TRAIN_DIR / "carrier", n_carrier, 0,
             config.TRAIN_MAX_SPEAKER, min_phoneme_count=60)

    # The same again for the pronunciations real speakers use (training only).
    spoken_txt = write_phrase_file(text_dir / "positive_spoken.txt",
                                   config.POSITIVE_SPELLINGS_SPOKEN)
    n_plain_sp = int(config.N_POSITIVE_TRAIN_SPOKEN * 0.7)
    generate(spoken_txt, config.POSITIVE_TRAIN_DIR / "plain_spoken", n_plain_sp, 0,
             config.TRAIN_MAX_SPEAKER)
    generate(spoken_txt, config.POSITIVE_TRAIN_DIR / "carrier_spoken",
             config.N_POSITIVE_TRAIN_SPOKEN - n_plain_sp, 0,
             config.TRAIN_MAX_SPEAKER, min_phoneme_count=60)

    n_plain_ho = int(config.N_POSITIVE_HELDOUT * 0.7)
    generate(positive_txt, config.POSITIVE_HELDOUT_DIR / "plain", n_plain_ho,
             config.HELDOUT_MIN_SPEAKER, config.HELDOUT_MAX_SPEAKER)
    generate(positive_txt, config.POSITIVE_HELDOUT_DIR / "carrier",
             config.N_POSITIVE_HELDOUT - n_plain_ho, config.HELDOUT_MIN_SPEAKER,
             config.HELDOUT_MAX_SPEAKER, min_phoneme_count=60)

    # --- adversarial negatives -------------------------------------------- #
    generate(adversarial_txt, config.ADVERSARIAL_DIR / "train",
             config.N_ADVERSARIAL_TRAIN, 0, config.TRAIN_MAX_SPEAKER)
    generate(adversarial_txt, config.ADVERSARIAL_DIR / "heldout",
             config.N_ADVERSARIAL_HELDOUT, config.HELDOUT_MIN_SPEAKER,
             config.HELDOUT_MAX_SPEAKER)

    print("\nStep 2 complete.")
    for d in (config.POSITIVE_TRAIN_DIR, config.POSITIVE_HELDOUT_DIR,
              config.ADVERSARIAL_DIR / "train", config.ADVERSARIAL_DIR / "heldout"):
        print(f"  {str(d.relative_to(config.WORK_DIR)):28s} {count_wavs(d):6d} wav")


if __name__ == "__main__":
    main()
