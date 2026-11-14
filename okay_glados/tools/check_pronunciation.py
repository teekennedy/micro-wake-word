#!/usr/bin/env python3
"""Verify that every configured spelling actually sounds like "Okay GLaDOS".

Run this before spending hours on training whenever you change
``config.POSITIVE_SPELLINGS``.  Two independent checks:

1. eSpeak phonemes -- what Piper will actually be asked to say.
2. Round-trip -- generate audio, transcribe it back with Whisper, and see
   whether a listener-equivalent hears the wake word.

Both run in the piper virtualenv:

    .venv-piper/bin/python okay_glados/tools/check_pronunciation.py

Add --no-asr to skip the Whisper round trip (much faster, no model download).
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# Spellings that are known-wrong, kept here so the check demonstrates the
# failures as well as the fix.  "Okay glahdohs" is included because it
# phonemises identically to "Okay glados" -- a tempting-looking variant that is
# exactly the pronunciation we are avoiding.
KNOWN_BAD = ["Okay GLaDOS", "Okay glados", "Okay glahdohs"]

# A transcription counts as correct if it contains "okay"/"ok" followed by
# something that starts with "gl" and is not broken into letters.  Whisper
# cannot reliably distinguish GLAD-ohs from GLAH-dos, so this is only the
# coarse check; the phoneme assertion below is the precise one.
GOOD = re.compile(r"\b(ok|okay|o\.?k\.?)\b[\s,.-]*gl", re.IGNORECASE)


def phonemes(text):
    from piper_phonemize import phonemize_espeak
    return " ".join("".join(s) for s in phonemize_espeak(text, "en-us"))


def generate(text, out_dir: Path, n: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(config.PIPER_DIR / "generate_samples.py"), text,
         "--model", str(config.PIPER_DIR / "models" / "en_US-libritts_r-medium.pt"),
         "--max-samples", str(n), "--batch-size", str(n),
         "--max-speakers", str(config.TRAIN_MAX_SPEAKER),
         "--output-dir", str(out_dir)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return sorted(out_dir.glob("*.wav"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--no-asr", action="store_true")
    ap.add_argument("--whisper-model", default="small.en")
    args = ap.parse_args()

    spellings = [s for s, _ in config.POSITIVE_SPELLINGS]
    print(f'Target wake word: "{config.WAKE_WORD}"\n')
    print("eSpeak phonemisation")
    print("-" * 72)
    phoneme_errors = []
    for text in spellings + KNOWN_BAD:
        ph = phonemes(text)
        tag = "  (known bad)" if text in KNOWN_BAD else ""
        if text not in KNOWN_BAD:
            if config.REQUIRED_PHONEMES not in ph:
                phoneme_errors.append(
                    f"{text!r} is missing {config.REQUIRED_PHONEMES!r}")
            if config.FORBIDDEN_PHONEMES in ph:
                phoneme_errors.append(
                    f"{text!r} contains {config.FORBIDDEN_PHONEMES!r} -- that is "
                    f'the rejected "Okay glados" pronunciation')
        print(f"  {text!r:22} {ph}{tag}")

    if phoneme_errors:
        print("\nFAIL -- phoneme check:")
        for e in phoneme_errors:
            print(f"  {e}")
        raise SystemExit(1)
    print(f"\nPhoneme check passed: every spelling contains "
          f"{config.REQUIRED_PHONEMES!r} and none contains "
          f"{config.FORBIDDEN_PHONEMES!r}.")

    if args.no_asr:
        return

    asr = None
    try:
        import whisper
        print(f"\nLoading Whisper {args.whisper_model} ...")
        asr = whisper.load_model(args.whisper_model)
    except ImportError:
        print("\nWhisper not installed; skipping the round-trip check.")
        print("  uv pip install --python .venv-piper openai-whisper")
        return

    tmp = Path(tempfile.mkdtemp(prefix="pron-"))
    failures = []
    try:
        print("\nRound trip: generate with Piper, transcribe with Whisper")
        print("-" * 72)
        for text in spellings + KNOWN_BAD:
            wavs = generate(text, tmp / re.sub(r"\W+", "_", text), args.samples)
            heard = [asr.transcribe(str(w), language="en", fp16=False,
                                    temperature=0.0)["text"].strip() for w in wavs]
            ok = sum(bool(GOOD.search(h)) for h in heard)
            verdict = f"{ok}/{len(heard)} recognisable"
            print(f"  {text!r:22} {verdict}")
            for h in heard:
                print(f"       {h}")
            if text not in KNOWN_BAD and ok < len(heard) * 0.6:
                failures.append((text, verdict))
            if text in KNOWN_BAD and ok > len(heard) * 0.6:
                print("       NOTE: Whisper accepted this known-bad spelling. "
                      "Whisper hears the consonants, not the vowel -- the "
                      "phoneme check above is the one that matters.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failures:
        print("FAIL -- these spellings do not reliably sound like the wake word:")
        for text, verdict in failures:
            print(f"  {text!r}: {verdict}")
        raise SystemExit(1)
    print("PASS -- every configured spelling sounds like the wake word.")


if __name__ == "__main__":
    main()
