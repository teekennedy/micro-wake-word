#!/usr/bin/env python3
"""Step 1 -- fetch every external dataset the pipeline needs.

  * MIT room impulse responses            -> reverberation augmentation
  * AudioSet balanced-train slices        -> background-noise augmentation
  * AudioSet eval slice                   -> EXTRA held-out ambient test audio
  * a second AudioSet eval slice          -> ambient audio for checkpoint selection
  * Free Music Archive (xs)               -> background-music augmentation
  * kahrendt/microwakeword negative sets  -> pre-computed negative spectrograms

The negative spectrogram sets are 10 ms window-step features (verified: the
CHiME-6 training mmap holds 40.49 h of features at 10 ms, matching the corpus's
published 40 h), which is what the v2 ESPHome models use.

Everything is idempotent: rerunning skips work that is already on disk.
"""

import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import scipy.io.wavfile
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

HF = "https://huggingface.co/datasets"

# AudioSet is distributed as ~650 MB parquet shards of flac audio.
AUDIOSET_BACKGROUND_SHARDS = [f"data/bal_train/{i:02d}.parquet" for i in range(3)]
AUDIOSET_AMBIENT_SHARDS = [f"data/eval/{i:02d}.parquet" for i in range(8)]
# Disjoint from the test slice above; used only to pick checkpoints.
AUDIOSET_AMBIENT_VAL_SHARDS = [f"data/eval/{i:02d}.parquet" for i in range(8, 16)]

NEGATIVE_ZIPS = [
    "dinner_party.zip",       # CHiME-6 train (40 h x 2 array channels)
    "dinner_party_eval.zip",  # CHiME-6 dev+eval (validation_ambient) + DiPCo (testing_ambient)
    "no_speech.zip",          # FMA-medium, FSD50K, WHAM!
    "speech.zip",             # VOiCES close/mid/far
]


def sh(cmd, **kwargs):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def download(url: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    sh(["curl", "-fL", "-sS", "--retry", "5", "--retry-delay", "5", "-o", str(tmp), url])
    tmp.rename(dest)
    return dest


def write_wav16k(path: Path, array: np.ndarray):
    scipy.io.wavfile.write(path, 16000, (np.asarray(array) * 32767).astype(np.int16))


def done_marker(directory: Path) -> Path:
    return directory / ".complete"


# --------------------------------------------------------------------------- #


def fetch_mit_rirs():
    if done_marker(config.RIR_DIR).exists():
        print("MIT RIRs already present")
        return
    import datasets

    print("Downloading MIT environmental impulse responses ...")
    config.RIR_DIR.mkdir(parents=True, exist_ok=True)
    ds = datasets.load_dataset(
        "davidscripka/MIT_environmental_impulse_responses",
        split="train",
        streaming=True,
    )
    n = 0
    for row in tqdm(ds, desc="rirs"):
        name = Path(row["audio"]["path"]).name
        write_wav16k(config.RIR_DIR / name, row["audio"]["array"])
        n += 1
    done_marker(config.RIR_DIR).write_text(str(n))
    print(f"  {n} impulse responses")


def fetch_audioset(shards, out_dir: Path, label: str):
    """Decode AudioSet's parquet-of-flac shards straight to 16 kHz wav.

    We go through pyarrow + soundfile rather than `datasets`, because these
    shards were written with a newer `datasets` whose feature metadata the
    pinned 3.x reader rejects.
    """
    todo = [sh for sh in shards
            if not (out_dir / f".done_{sh.replace('/', '_')}").exists()]
    if not todo:
        print(f"AudioSet {label} already present ({len(shards)} shards)")
        return
    import io

    import soundfile as sf
    import soxr
    import pyarrow.parquet as pq

    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for shard in todo:
        local = download(
            f"{HF}/agkphysics/AudioSet/resolve/main/{shard}",
            config.DOWNLOAD_DIR / f"audioset_{shard.replace('/', '_')}",
        )
        pf = pq.ParquetFile(local)
        print(f"Converting {local.name} ({pf.metadata.num_rows} clips) -> 16 kHz wav ({label}) ...")
        for batch in tqdm(pf.iter_batches(batch_size=32, columns=["video_id", "audio"]),
                          total=(pf.metadata.num_rows + 31) // 32, desc=local.name):
            for row in batch.to_pylist():
                try:
                    audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
                except Exception:
                    continue
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sr != 16000:
                    audio = soxr.resample(audio, sr, 16000)
                if audio.size < 16000:
                    continue
                write_wav16k(out_dir / f"{row['video_id']}.wav", audio)
                n += 1
        local.unlink()
        (out_dir / f".done_{shard.replace('/', '_')}").write_text(str(n))
    done_marker(out_dir).write_text(str(n))
    print(f"  {n} new clips in {out_dir}")


def fetch_fma():
    if done_marker(config.BACKGROUND_FMA_DIR).exists():
        print("FMA already present")
        return
    import datasets

    archive = download(
        f"{HF}/mchl914/fma_xsmall/resolve/main/fma_xs.zip",
        config.DOWNLOAD_DIR / "fma_xs.zip",
    )
    raw = config.WORK_DIR / "fma_raw"
    if not raw.exists():
        raw.mkdir(parents=True)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(raw)

    config.BACKGROUND_FMA_DIR.mkdir(parents=True, exist_ok=True)
    mp3s = [str(p) for p in raw.glob("**/*.mp3")]
    print(f"Converting {len(mp3s)} FMA tracks -> 16 kHz wav ...")
    ds = datasets.Dataset.from_dict({"audio": mp3s}).cast_column(
        "audio", datasets.Audio(sampling_rate=16000)
    )
    n = 0
    for row in tqdm(ds, desc="fma"):
        name = Path(row["audio"]["path"]).stem + ".wav"
        write_wav16k(config.BACKGROUND_FMA_DIR / name, row["audio"]["array"])
        n += 1
    done_marker(config.BACKGROUND_FMA_DIR).write_text(str(n))
    shutil.rmtree(raw, ignore_errors=True)
    archive.unlink(missing_ok=True)
    print(f"  {n} music clips")


def fetch_negative_features():
    config.NEGATIVE_DIR.mkdir(parents=True, exist_ok=True)
    for name in NEGATIVE_ZIPS:
        target = config.NEGATIVE_DIR / name.replace(".zip", "")
        if done_marker(target).exists():
            print(f"  have {target.name}")
            continue
        archive = download(
            f"{HF}/kahrendt/microwakeword/resolve/main/{name}",
            config.DOWNLOAD_DIR / name,
        )
        print(f"Extracting {name} ...")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(config.NEGATIVE_DIR)
        # Drop the macOS resource-fork junk the archives carry.
        shutil.rmtree(config.NEGATIVE_DIR / "__MACOSX", ignore_errors=True)
        done_marker(target).write_text("ok")
        archive.unlink()  # reclaim the disk immediately
        print(f"  extracted {target}")


def main():
    config.WORK_DIR.mkdir(parents=True, exist_ok=True)
    config.DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    fetch_negative_features()
    fetch_mit_rirs()
    fetch_fma()
    fetch_audioset(AUDIOSET_BACKGROUND_SHARDS, config.BACKGROUND_AUDIOSET_DIR, "backgrounds")
    fetch_audioset(AUDIOSET_AMBIENT_SHARDS, config.AMBIENT_EXTRA_DIR, "held-out ambient")
    fetch_audioset(AUDIOSET_AMBIENT_VAL_SHARDS, config.AMBIENT_VAL_DIR, "validation ambient")

    print("\nStep 1 complete.")
    for d in [
        config.NEGATIVE_DIR,
        config.RIR_DIR,
        config.BACKGROUND_FMA_DIR,
        config.BACKGROUND_AUDIOSET_DIR,
        config.AMBIENT_EXTRA_DIR,
        config.AMBIENT_VAL_DIR,
    ]:
        size = sum(f.stat().st_size for f in d.glob("**/*") if f.is_file())
        print(f"  {d.name:24s} {size / 1e9:8.2f} GB")


if __name__ == "__main__":
    main()
