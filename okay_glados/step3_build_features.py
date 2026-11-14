#!/usr/bin/env python3
"""Step 3 -- turn audio clips into the spectrogram feature sets used for training.

Produces mmap-ninja feature folders laid out the way microwakeword.data expects:

    <features_dir>/training/*_mmap
    <features_dir>/validation/*_mmap
    <features_dir>/testing/*_mmap
    <features_dir>/testing_ambient/*_mmap

Four feature sets are built here:

  positive/           synthetic wake word          (truth=True)
  positive_recorded/  your hand-recorded wake word (truth=True)  -- optional
  adversarial/        near-miss TTS phrases        (truth=False)
  ambient_extra/      held-out AudioSet audio      (truth=False, testing_ambient)

The pre-computed negative sets downloaded in step 1 are used as-is.

Augmentation (background noise, room impulse responses, EQ, gain, pitch) is
applied here, once, and baked into the stored spectrograms.  Each source clip is
rendered `repeat` times with different random augmentation, and `slide_frames`
one-frame-shifted copies are emitted per render so that training in
non-streaming mode matches what the streaming model sees on device.

Work is sharded across processes because a single generator is CPU-bound; the
loader globs `**/*_mmap`, so several shard folders in one split are equivalent
to one big one.
"""

import argparse
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

AUDIO_SUFFIXES = {".wav", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".aac", ".wma", ".aiff", ".aif"}
N_WORKERS = max(1, min(8, (os.cpu_count() or 4) - 2))

# Augmentation probabilities.  AddBackgroundNoise and RIR are the two that
# matter most for far-field wake word robustness.
AUGMENT_PROBABILITIES = {
    "SevenBandParametricEQ": 0.25,
    "TanhDistortion": 0.15,
    "PitchShift": 0.25,
    "BandStopFilter": 0.15,
    "AddColorNoise": 0.25,
    "AddBackgroundNoise": 0.85,
    "Gain": 1.0,
    "GainTransition": 0.25,
    "RIR": 0.6,
}

# A gentler profile used only for the "normal use" test set: an ordinary room,
# an ordinary speaking volume, no distortion or notch filtering.
MILD_AUGMENT_PROBABILITIES = {
    "SevenBandParametricEQ": 0.15,
    "TanhDistortion": 0.0,
    "PitchShift": 0.0,
    "BandStopFilter": 0.0,
    "AddColorNoise": 0.15,
    "AddBackgroundNoise": 0.5,
    "Gain": 1.0,
    "GainTransition": 0.0,
    "RIR": 0.35,
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def list_audio(root: Path, recursive=True):
    if not root.exists():
        return []
    it = root.rglob("*") if recursive else root.glob("*")
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES)


def stage_shards(paths, staging_root: Path, n_shards: int):
    """Symlink `paths` into n_shards sibling folders and return their paths."""
    shutil.rmtree(staging_root, ignore_errors=True)
    shards = []
    n_shards = max(1, min(n_shards, len(paths)))
    for i in range(n_shards):
        d = staging_root / f"shard{i:02d}"
        d.mkdir(parents=True)
        shards.append(d)
    for idx, src in enumerate(paths):
        dest = shards[idx % n_shards] / f"{idx:07d}{src.suffix.lower()}"
        dest.symlink_to(src.resolve())
    return shards


def deterministic_split(paths, fractions):
    """Split a sorted path list into named buckets, deterministically.

    Interleaving by index rather than shuffling keeps the buckets balanced even
    for very small recording sets, and keeps the assignment stable when new
    files are added at the end.
    """
    names = list(fractions)
    weights = [fractions[n] for n in names]
    total = sum(weights)
    buckets = {n: [] for n in names}
    # Build a repeating pattern of bucket names matching the requested ratio.
    pattern = []
    for name, w in zip(names, weights):
        pattern.extend([name] * max(1, round(w / total * 10)))
    for i, p in enumerate(paths):
        buckets[pattern[i % len(pattern)]].append(p)
    return buckets


def convert_to_wav16k(paths, out_dir: Path):
    """Normalise arbitrary user recordings to 16 kHz mono wav with ffmpeg."""
    out_dir.mkdir(parents=True, exist_ok=True)
    converted = []
    for i, src in enumerate(paths):
        dest = out_dir / f"{i:05d}_{src.stem[:40]}.wav"
        if not dest.exists():
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
                 "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(dest)],
                check=True,
            )
        converted.append(dest)
    return converted


def session_of(path: Path) -> str:
    """Segmented clips are named <session>_<n> or <session>_bg<n>."""
    return re.sub(r"_(bg)?\d{4}$", "", Path(path).stem)


LEAD_IN_S = 1.5


def add_lead_in(seg_dir: Path):
    """Re-cut each segmented wake word with the session audio that preceded it.

    The segmenter cuts takes tightly (0.12 s of padding).  Rendered into a
    3.2 s window that puts digital silence, then an abrupt step up to the
    room's noise floor, right before every real wake word -- and models trained
    on that fired three times as often in the first two seconds of a test clip
    as anywhere else: they had learned the step.  On the device the wake word
    is always preceded by continuous room sound, so give it that: up to
    LEAD_IN_S of what was actually recorded before the take (stopping at the
    previous take).  Returns the new clip paths, in the same order.
    """
    import numpy as np
    import soundfile as sf

    sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
    from segment_recordings import load16k

    rows = {}
    for line in (seg_dir / "manifest.tsv").read_text().splitlines()[1:]:
        name, start, end, decision = line.split("\t")[:4]
        if decision.startswith("positive"):
            rows.setdefault(name, []).append((float(start), float(end)))
    sessions = {}
    for src in list_audio(config.RECORDINGS_DIR):
        if src.name in rows:
            tag = re.sub(r"[^A-Za-z0-9]+", "_", str(
                src.relative_to(config.RECORDINGS_DIR).with_suffix(""))).strip("_")
            sessions[tag] = (src, sorted(rows[src.name]))

    out_dir = seg_dir / "positive_leadin"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    result, lead_total = [], 0.0
    for tag, (src, takes) in sorted(sessions.items()):
        audio = load16k(src)
        clips = sorted((seg_dir / "positive").glob(f"{tag}_[0-9][0-9][0-9][0-9].wav"))
        if len(clips) != len(takes):
            raise SystemExit(f"{src.name}: {len(clips)} clips but {len(takes)} manifest rows "
                             "-- rerun tools/segment_recordings.py")
        prev_end = 0
        for clip_path, (start, end) in zip(clips, takes):
            clip, _ = sf.read(clip_path, dtype="float32")
            a = int(round(start * 16000))
            n = min(len(clip), 4000)
            if np.abs(audio[a:a + n] - clip[:n]).max() > 1e-3:
                raise SystemExit(f"{clip_path.name} does not match {src.name} at {start:.2f} s")
            lo = max(prev_end, a - int(LEAD_IN_S * 16000), 0)
            lead = audio[lo:a].copy()
            fade = min(len(lead), 800)           # 50 ms fade-in, no click
            lead[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
            dest = out_dir / clip_path.name
            sf.write(dest, np.concatenate([lead, clip]), 16000, subtype="PCM_16")
            result.append(dest)
            lead_total += len(lead) / 16000
            prev_end = a + len(clip)
    print(f"  lead-in added to {len(result)} takes "
          f"(mean {lead_total / max(1, len(result)):.2f} s)")
    return sorted(result)


def normalise(positives, session_negatives, out_dir: Path, peak_dbfs: float = -3.0):
    """Bring hand recordings up to the level of the TTS clips.

    Phone recordings sit ~18 dB below the near-full-scale TTS clips, and the
    augmenter's -35..0 dB gain then pushes them toward silence.  One gain is
    applied per *session* (from the loud end of its wake-word takes) to the
    positives and to the negatives cut from the same session, so the level of
    the background relative to the voice stays as recorded and cannot become
    a cue.  Single-take files are their own session.
    """
    import numpy as np
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    peaks = {}
    for src in positives:
        audio, _ = sf.read(src, dtype="float32")
        peaks.setdefault(session_of(src), []).append(float(np.abs(audio).max()))
    target = 10 ** (peak_dbfs / 20)
    gains = {k: target / max(np.percentile(v, 90), 1e-4) for k, v in peaks.items()}

    def apply(paths, sub):
        result = []
        for src in paths:
            audio, sr = sf.read(src, dtype="float32")
            gain = gains.get(session_of(src))
            if gain is None:
                continue  # a negative from a session with no wake words
            dest = out_dir / sub / Path(src).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            sf.write(dest, np.clip(audio * gain, -1, 1), sr, subtype="PCM_16")
            result.append(dest)
        return sorted(result)

    return apply(positives, "positive"), apply(session_negatives, "negative")


# --------------------------------------------------------------------------- #
# worker
# --------------------------------------------------------------------------- #


def _build_shard(job):
    """Runs in a subprocess: render one shard folder into one *_mmap folder."""
    # Imported here so each worker gets its own TF/audiomentations state.
    from mmap_ninja.ragged import RaggedMmap
    from microwakeword.audio.augmentation import Augmentation
    from microwakeword.audio.clips import Clips
    from microwakeword.audio.spectrograms import SpectrogramGeneration

    clips = Clips(
        input_directory=job["shard_dir"],
        file_pattern="*.wav",
        max_clip_duration_s=job["max_clip_duration_s"],
        remove_silence=job["remove_silence"],
        trim_zeros=True,
        random_split_seed=None,
    )
    if len(clips.clips) == 0:
        return job["out_dir"], 0

    augmenter = None
    if job["augment"]:
        augmenter = Augmentation(
            augmentation_duration_s=config.AUGMENTATION_DURATION_S,
            augmentation_probabilities=(
                MILD_AUGMENT_PROBABILITIES if job["mild"] else AUGMENT_PROBABILITIES),
            impulse_paths=job["impulse_paths"],
            background_paths=job["background_paths"],
            background_min_snr_db=job["background_min_snr_db"],
            background_max_snr_db=job["background_max_snr_db"],
            min_gain_db=-12 if job["mild"] else -35,
            max_gain_db=0,
            min_jitter_s=0.195,
            max_jitter_s=0.205,
        )

    spectrograms = SpectrogramGeneration(
        clips=clips,
        augmenter=augmenter,
        step_ms=config.WINDOW_STEP_MS,
        slide_frames=job["slide_frames"],
    )

    def long_enough(generator):
        """Drop spectrograms the model cannot consume.

        Real-world clips are occasionally digital silence, and trimming zeros
        from those leaves nothing behind.  A short or empty spectrogram makes
        microwakeword.data.fixed_length_spectrogram raise, so filter here.
        """
        for spectrogram in generator:
            if spectrogram.ndim == 2 and spectrogram.shape[0] >= job["min_frames"]:
                yield spectrogram

    out_dir = Path(job["out_dir"])
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    RaggedMmap.from_generator(
        out_dir=str(out_dir),
        sample_generator=long_enough(
            spectrograms.spectrogram_generator(repeat=job["repeat"])),
        batch_size=64,
        verbose=False,
    )
    return job["out_dir"], len(RaggedMmap(str(out_dir)))


def build_set(name, paths, features_dir: Path, split: str, repeat: int,
              slide_frames, augment=True, remove_silence=False,
              max_clip_duration_s=None, background_min_snr_db=-5,
              background_max_snr_db=12, mild=False, n_workers=N_WORKERS):
    if not paths:
        print(f"  [{name}/{split}] no source audio, skipping")
        return 0
    staging = config.WORK_DIR / "staging" / name / split
    shards = stage_shards(paths, staging, n_workers)

    impulse_paths = [str(config.RIR_DIR)] if config.RIR_DIR.exists() else []
    background_paths = [
        str(p) for p in (config.BACKGROUND_FMA_DIR, config.BACKGROUND_AUDIOSET_DIR)
        if p.exists() and any(p.glob("*.wav"))
    ]
    if augment and not background_paths:
        raise SystemExit("No background audio found -- run step1_download_data.py first")

    jobs = [
        {
            "shard_dir": str(d),
            "out_dir": str(features_dir / split / f"{name}_{d.name}_mmap"),
            "repeat": repeat,
            "slide_frames": slide_frames,
            "augment": augment,
            "remove_silence": remove_silence,
            "max_clip_duration_s": max_clip_duration_s,
            "impulse_paths": impulse_paths,
            "background_paths": background_paths,
            "background_min_snr_db": background_min_snr_db,
            "background_max_snr_db": background_max_snr_db,
            "min_frames": config.SPECTROGRAM_LENGTH,
            "mild": mild,
        }
        for d in shards
    ]
    print(f"  [{name}/{split}] {len(paths)} clips x{repeat} across {len(jobs)} workers ...",
          flush=True)
    total = 0
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        for out_dir, n in pool.imap_unordered(_build_shard, jobs):
            total += n
    shutil.rmtree(staging, ignore_errors=True)
    print(f"  [{name}/{split}] {total} spectrograms", flush=True)
    return total


# --------------------------------------------------------------------------- #
# the four feature sets
# --------------------------------------------------------------------------- #


def build_positive():
    print("Positive (synthetic) features")
    train = list_audio(config.POSITIVE_TRAIN_DIR)
    spoken = [p for p in train if p.parent.name.endswith("_spoken")]
    build_set("positive", [p for p in train if not p.parent.name.endswith("_spoken")],
              config.FEATURES_POSITIVE, "training", repeat=2, slide_frames=5)
    build_positive_spoken(spoken)
    heldout = list_audio(config.POSITIVE_HELDOUT_DIR)
    buckets = deterministic_split(heldout, {"validation": 0.5, "testing": 0.5})
    build_set("positive", buckets["validation"], config.FEATURES_POSITIVE,
              "validation", repeat=1, slide_frames=5)
    build_set("positive", buckets["testing"], config.FEATURES_POSITIVE,
              "testing", repeat=1, slide_frames=1)
    # The same held-out audio again, but rendered as ordinary room speech.  The
    # harsh set above (down to -5 dB SNR and -35 dB gain) contains utterances a
    # person could not hear either; this one is what day-to-day use looks like.
    build_set("positive_clean", buckets["testing"], config.FEATURES_POSITIVE_CLEAN,
              "testing", repeat=1, slide_frames=1, mild=True,
              background_min_snr_db=10, background_max_snr_db=30)


def build_positive_spoken(paths=None):
    """The real-speaker pronunciations (config.POSITIVE_SPELLINGS_SPOKEN).

    A separate shard family inside positive/training, so it can be (re)built
    with ``--only spoken`` without touching the 14 GB original set.
    """
    if paths is None:
        paths = [p for p in list_audio(config.POSITIVE_TRAIN_DIR)
                 if p.parent.name.endswith("_spoken")]
    build_set("positive_spoken", paths, config.FEATURES_POSITIVE,
              "training", repeat=2, slide_frames=5)


def build_adversarial():
    print("Adversarial (near-miss TTS) features")
    build_set("adversarial", list_audio(config.ADVERSARIAL_DIR / "train"),
              config.FEATURES_ADVERSARIAL, "training", repeat=2, slide_frames=5)
    heldout = list_audio(config.ADVERSARIAL_DIR / "heldout")
    buckets = deterministic_split(heldout, {"validation": 0.5, "testing": 0.5})
    build_set("adversarial", buckets["validation"], config.FEATURES_ADVERSARIAL,
              "validation", repeat=1, slide_frames=5)
    build_set("adversarial", buckets["testing"], config.FEATURES_ADVERSARIAL,
              "testing", repeat=1, slide_frames=1)


def duration_s(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return float(out.stdout.strip() or 0)


def build_recorded():
    print("Recorded (hand-recorded) features")
    all_pos = [p for p in list_audio(config.RECORDINGS_DIR)
               if "hard_negatives" not in p.parts]
    all_neg = list_audio(config.RECORDINGS_DIR / "hard_negatives")
    if not all_pos and not all_neg:
        print("  okay_glados/recordings/ is empty -- skipping.")
        print("  Add wav/m4a/mp3 files there and rerun with --only recorded.")
        return

    # Long files are sessions of many takes (or an hour of room audio).  Used
    # whole, the augmenter would keep only their last 3.2 s, so they must go
    # through tools/segment_recordings.py first.
    sessions = [p for p in all_pos if duration_s(p) > config.MAX_SINGLE_TAKE_S]
    raw = [p for p in all_pos if p not in sessions]
    neg_sessions = [p for p in all_neg if duration_s(p) > config.MAX_SINGLE_TAKE_S]
    negatives = [p for p in all_neg if p not in neg_sessions]
    seg = config.SEGMENTED_DIR
    if (sessions or neg_sessions) and not (seg / "manifest.tsv").exists():
        raise SystemExit(
            f"{len(sessions) + len(neg_sessions)} recording(s) are longer than "
            f"{config.MAX_SINGLE_TAKE_S:.0f} s -- run tools/segment_recordings.py first")
    seg_pos = add_lead_in(seg) if sessions else []
    seg_neg = (list_audio(seg / "negative") + list_audio(seg / "background")
               if sessions else [])
    home_train = list_audio(seg / "home_train") if neg_sessions else []
    home_test = list_audio(seg / "home_test") if neg_sessions else []
    print(f"  single takes: {len(raw)}; from sessions: {len(seg_pos)} wake words, "
          f"{len(seg_neg)} speech/background negatives, {len(home_train)} room-audio chunks, "
          f"{len(home_test)} held-out room-audio blocks")

    shutil.rmtree(config.FEATURES_POSITIVE_RECORDED, ignore_errors=True)
    shutil.rmtree(config.FEATURES_HOME_AMBIENT, ignore_errors=True)
    for stale in (config.FEATURES_ADVERSARIAL / "training").glob("recorded_negative_*_mmap"):
        shutil.rmtree(stale)
    if raw or seg_pos:
        out16k = config.WORK_DIR / "recordings_16k"
        shutil.rmtree(out16k, ignore_errors=True)
        wavs = sorted(convert_to_wav16k(raw, out16k) + seg_pos)
        wavs, seg_neg = normalise(wavs, seg_neg, out16k / "normalised")
        print(f"  {len(wavs)} recordings found")
        if len(wavs) < 10:
            print("  WARNING: fewer than 10 recordings; using all of them for "
                  "training and reporting no recorded-speech test metric.")
            buckets = {"training": wavs, "validation": [], "testing": []}
        else:
            buckets = deterministic_split(
                wavs, {"training": 0.6, "validation": 0.2, "testing": 0.2})
        # No VAD trimming (remove_silence): it would strip the lead-in that
        # add_lead_in() just put there.
        # Few clips, so render many augmented variants of each -- but not so
        # many that the model can memorise each clip's background.
        repeat = max(4, min(100, round(2000 / max(1, len(buckets["training"])))))
        build_set("recorded", buckets["training"], config.FEATURES_POSITIVE_RECORDED,
                  "training", repeat=repeat, slide_frames=5)
        # Held-out recordings already carry real room acoustics and noise, so
        # they are rendered with the mild "normal use" profile, matching how
        # the synthetic normal-use test set is built.
        build_set("recorded", buckets["validation"], config.FEATURES_POSITIVE_RECORDED,
                  "validation", repeat=8, slide_frames=5,
                  mild=True, background_min_snr_db=10, background_max_snr_db=30)
        build_set("recorded", buckets["testing"], config.FEATURES_POSITIVE_RECORDED,
                  "testing", repeat=8, slide_frames=1,
                  mild=True, background_min_snr_db=10, background_max_snr_db=30)

    if negatives or seg_neg or home_train:
        out16k = config.WORK_DIR / "recordings_neg_16k"
        shutil.rmtree(out16k, ignore_errors=True)
        wavs = convert_to_wav16k(negatives, out16k) + seg_neg + home_train
        print(f"  {len(wavs)} hard-negative clips found")
        repeat = max(4, min(200, round(4000 / len(wavs))))
        build_set("recorded_negative", wavs, config.FEATURES_ADVERSARIAL,
                  "training", repeat=repeat, slide_frames=5)

    if home_test:
        # Like ambient_extra: unaugmented, streamed whole in step 5 to count
        # false accepts per hour in the room the device actually lives in.
        build_set("home_ambient", home_test, config.FEATURES_HOME_AMBIENT,
                  "testing_ambient", repeat=1, slide_frames=None, augment=False)


def build_ambient_extra():
    print("Extra held-out ambient (AudioSet eval) features")
    paths = list_audio(config.AMBIENT_EXTRA_DIR)
    if not paths:
        print("  none found, skipping")
        return
    # No augmentation and no sliding: these are streamed whole to count false
    # accepts per hour, so they must stay pristine, real-world audio.
    build_set("ambient_extra", paths, config.FEATURES_AMBIENT_EXTRA, "testing_ambient",
              repeat=1, slide_frames=None, augment=False)
    # A disjoint slice for checkpoint selection (and the trainer's own
    # validation pass), so music and TV false accepts are visible before the
    # test set is touched.
    build_set("ambient_val", list_audio(config.AMBIENT_VAL_DIR),
              config.FEATURES_AMBIENT_VAL, "validation_ambient",
              repeat=1, slide_frames=None, augment=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["positive", "spoken", "adversarial", "recorded", "ambient"],
                    help="build only these feature sets")
    args = ap.parse_args()
    todo = args.only or ["positive", "adversarial", "recorded", "ambient"]

    if "positive" in todo:
        build_positive()
    elif "spoken" in todo:
        build_positive_spoken()
    if "adversarial" in todo:
        build_adversarial()
    if "recorded" in todo:
        build_recorded()
    if "ambient" in todo:
        build_ambient_extra()

    print("\nStep 3 complete.")
    for d in [config.FEATURES_POSITIVE, config.FEATURES_POSITIVE_CLEAN,
              config.FEATURES_POSITIVE_RECORDED,
              config.FEATURES_ADVERSARIAL, config.FEATURES_AMBIENT_EXTRA,
              config.FEATURES_AMBIENT_VAL,
              config.FEATURES_HOME_AMBIENT]:
        if d.exists():
            size = sum(f.stat().st_size for f in d.glob("**/*") if f.is_file())
            print(f"  {d.name:22s} {size / 1e9:7.2f} GB")


if __name__ == "__main__":
    main()
