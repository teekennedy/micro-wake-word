#!/usr/bin/env python3
"""Split long hand-recorded sessions into the clips step 3 trains on.

A phone recording of someone saying "Okay GLaDOS" forty times, with commands,
chatter and TV in between, cannot be used as one clip: the augmenter keeps only
the *last* 3.2 s of anything longer.  This tool cuts such sessions apart.

Positive sessions (any file in recordings/ longer than
config.MAX_SINGLE_TAKE_S, outside hard_negatives/):

  1. An energy VAD finds speech and groups it into chunks at pauses.
  2. Each chunk is transcribed *on its own and with no prompt*, so Whisper's
     label is honest rather than primed to hear "GLaDOS".
  3. "okay <one word>" chunks are wake words.  Whisper spells GLaDOS a dozen
     ways ("Gladys", "Klaus", "Blados"...), so the second word is not
     checked: inside a wake-word session the speaker is saying the wake word.
  4. "okay GLaDOS, turn on the lights" is cut at the pause after the wake
     word; long back-to-back runs are split at energy dips.  Every cut is
     re-transcribed and kept only if it still reads as the wake word alone.
  5. The gaps between takes (TV, music, room tone from the same phone and
     room) become negatives, so the model cannot learn "this background
     means wake word".  Each gap chunk is transcribed and dropped if it
     contains anything okay/GLaDOS-like.
  6. Chunks of four or more words with nothing okay/gl-like in them become
     extra negatives (commands, conversation, TV).  Short or garbled chunks
     -- "you", "Thank you", "Kick gloves" -- are dropped from both classes:
     they are usually a quiet wake word Whisper failed on, and training those
     as negatives would teach the model to miss exactly the hard cases.

Hard-negative sessions (recordings/hard_negatives/):

  * Cut into 5-minute blocks; every third block is held out as a home
    false-accept test set (home_test/, streamed whole in step 5), the rest are
    cut into 3 s training chunks (home_train/).
  * The whole file is transcribed, and any training chunk within 5 s of a
    wake-word-like token (gl*/cl*/kl* after "okay", or any GLaDOS spelling) is
    dropped, so a stray "okay GLaDOS" can never be trained as a negative.

Requires pywhispercpp and a whisper.cpp ggml model, e.g. large-v3-turbo:

    uv run --no-project --with pywhispercpp --with soundfile --with numpy \\
        python okay_glados/tools/segment_recordings.py --model ggml-large-v3-turbo.bin

Writes config.SEGMENTED_DIR and a manifest.tsv listing every chunk, its
transcript and what was done with it -- read it before training.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

SR = 16000
HOP = 160                # 10 ms analysis frames
GAP_S = 0.45             # a pause this long separates two utterances
MIN_ISLAND_S = 0.12
PAD_S = 0.12             # context kept either side of a clip
MAX_WAKE_FRAMES = 230    # a single take longer than 2.3 s is not a single take
NEG_CHUNK_S = 3.0
NEG_BLOCK_S = 300.0
HOLDOUT_EVERY = 3        # every 3rd 5-minute block of room audio is held out
EXCLUSION_S = 5.0
AUDIO_SUFFIXES = {".wav", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".aac", ".wma",
                  ".aiff", ".aif"}


# --------------------------------------------------------------------------- #
# audio helpers
# --------------------------------------------------------------------------- #


def load16k(path: Path) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
        check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def frames_db(x):
    n = len(x) // HOP
    f = x[: n * HOP].reshape(n, HOP)
    db = 10 * np.log10(np.mean(f ** 2, axis=1) + 1e-10)
    return np.convolve(db, np.ones(5) / 5, mode="same")


def speech_islands(db, thr):
    on = np.concatenate(([False], db > thr, [False]))
    edges = np.flatnonzero(np.diff(on.astype(np.int8)))
    runs = [[int(a), int(b)] for a, b in zip(edges[::2], edges[1::2])]
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < 15:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [r for r in merged if r[1] - r[0] >= MIN_ISLAND_S * 100]


def group(isl, gap_s):
    chunks = []
    for a, b in isl:
        if chunks and a - chunks[-1][-1][1] < gap_s * 100:
            chunks[-1].append([a, b])
        else:
            chunks.append([[a, b]])
    return chunks


def split_long(isl, gap_s):
    """Recursively re-split chunks longer than 2.2 s on ever-shorter pauses."""
    out = []
    for ch in group(isl, gap_s):
        if ch[-1][1] - ch[0][0] > 220 and gap_s > 0.1 and len(ch) > 1:
            out.extend(split_long(ch, gap_s * 0.6))
        else:
            out.append(ch)
    return out


def write(path: Path, audio):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, SR, subtype="PCM_16")


# --------------------------------------------------------------------------- #
# transcript classification
# --------------------------------------------------------------------------- #

OKAY = ("ok", "okay", "okey", "okie")


def words_of(text):
    return re.sub(r"[^a-z ]", " ", text.lower()).split()


def is_wake_strict(w):
    return 2 <= len(w) <= 3 and w[0] in OKAY and re.match(r"^(gl|cl|kl|gr|g$)", w[1])


def is_wake(w):
    """Inside a wake-word session: okay + exactly one other word."""
    return bool(is_wake_strict(w)) or (len(w) == 2 and w[0] in OKAY)


def safe_negative(w):
    return len(w) >= 4 and not any(
        x in OKAY or re.match(r"^(gl|cl|kl)", x) for x in w)


class Transcriber:
    def __init__(self, model_path):
        from pywhispercpp.model import Model
        self.short = Model(model_path, language="en", print_progress=False,
                           print_realtime=False, no_context=True, single_segment=True)
        self.model_path = model_path

    def __call__(self, audio):
        pad = np.zeros(int(0.4 * SR), dtype=np.float32)
        segs = self.short.transcribe(np.concatenate([pad, audio.astype(np.float32), pad]))
        return " ".join(s.text.strip() for s in segs).strip()

    def words_with_times(self, audio):
        from pywhispercpp.model import Model
        m = Model(self.model_path, language="en", print_progress=False,
                  print_realtime=False, token_timestamps=True, max_len=1,
                  split_on_word=True)
        return [{"t0": s.t0 / 100, "t1": s.t1 / 100, "w": s.text.strip()}
                for s in m.transcribe(audio.astype(np.float32)) if s.text.strip()]


# --------------------------------------------------------------------------- #
# positive sessions
# --------------------------------------------------------------------------- #


def scan_chunk(asr, x, db, a, b):
    """Find successive wake words inside a long chunk, cutting at energy dips."""
    body = db[a:b]
    dips = sorted({a, b, *[a + i for i in range(5, len(body) - 5)
                           if body[i] == body[max(0, i - 12): i + 13].min()]})
    found, i = [], 0
    while i < len(dips) - 1:
        hit = None
        for j in range(i + 1, len(dips)):
            s0, e0 = dips[i], dips[j]
            if e0 - s0 < 45:
                continue
            if e0 - s0 > 210:
                break
            text = asr(x[s0 * HOP: (e0 + 3) * HOP])
            if is_wake(words_of(text)):
                hit = (s0, min(e0 + 3, b), text, j)
                break
            if len(words_of(text)) > 3:
                break
        if hit:
            found.append(hit[:3])
            i = hit[3]
        else:
            i += 1
    return found


def cut_command(asr, x, db, ch, lo, a, b):
    """'okay GLaDOS, turn on the lights' -> shortest prefix that is the wake word."""
    body = db[a:b]
    gaps = [e for (_, e) in ch[:-1]]
    dips = [a + i for i in range(50, min(len(body) - 10, 200))
            if body[i] == body[max(0, i - 8): i + 9].min()]
    for c in sorted(set(gaps + dips)):
        if not 45 <= c - a <= 200:
            continue
        sub = x[lo * HOP: min(c + 6, b) * HOP]
        text = asr(sub)
        if is_wake_strict(words_of(text)):
            return sub, text
    return None, None


def segment_positive_session(asr, path: Path, out: Path, tag: str):
    x = load16k(path)
    db = frames_db(x)
    floor = np.percentile(db, 10)
    thr = floor + max(6.0, 0.22 * (np.percentile(db, 97) - floor))
    chunks = split_long(speech_islands(db, thr), GAP_S)

    res, rows = [], []  # res: [lo, hi, decision, text, audio]
    for ci, ch in enumerate(chunks):
        a, b = ch[0][0], ch[-1][1]
        prev_end = chunks[ci - 1][-1][1] if ci else 0
        next_start = chunks[ci + 1][0][0] if ci + 1 < len(chunks) else len(db)
        lo = max(prev_end + 3, a - int(PAD_S * 100))
        hi = min(next_start - 3, b + int(PAD_S * 100))
        seg = x[lo * HOP: hi * HOP]
        text = asr(seg)
        w = words_of(text)
        if is_wake(w) and hi - lo <= MAX_WAKE_FRAMES:
            res.append([lo, hi, "positive", text, seg])
        elif hi - lo > MAX_WAKE_FRAMES and any(t in OKAY for t in w):
            found = scan_chunk(asr, x, db, a, b)
            res.append([lo, hi, f"split into {len(found)}", text, None])
            res.extend([s0, e0, "positive (split)", t, x[s0 * HOP: e0 * HOP]]
                       for s0, e0, t in found)
        elif w[:1] and w[0] in OKAY and len(w) > 2:
            clip, t2 = cut_command(asr, x, db, ch, lo, a, b)
            if clip is not None:
                res.append([lo, hi, "positive (command cut)", f"{text} => {t2}", clip])
            else:
                res.append([lo, hi, "skip: could not isolate", text, None])
        elif safe_negative(w):
            res.append([lo, hi, "negative", text, seg])
        else:
            res.append([lo, hi, "skip: ambiguous", text, None])

    # A slow "Okay. ... Gladys." can straddle two chunks: rejoin it.
    for i in range(len(res) - 1):
        w1, w2 = words_of(res[i][3]), words_of(res[i + 1][3])
        if (len(w1) == 1 and w1[0] in OKAY and len(w2) == 1
                and re.match(r"^(gl|cl|kl)", w2[0]) and res[i + 1][0] - res[i][1] < 120
                and res[i][4] is None and res[i + 1][4] is None):
            lo, hi = res[i][0], res[i + 1][1]
            res[i] = [lo, hi, "positive (rejoined)", f"{res[i][3]} + {res[i + 1][3]}",
                      x[lo * HOP: hi * HOP]]
            res[i + 1][2] = "skip: rejoined above"

    # The gaps between takes -- TV, game audio, room tone -- recorded on the
    # same phone in the same room as the positives.  Without these as
    # negatives the model can learn "this background means wake word".
    busy = np.zeros(len(db), dtype=bool)
    for ch in chunks:
        busy[max(0, ch[0][0] - 50): ch[-1][1] + 50] = True
    for lo, hi, *_ in res:
        busy[max(0, lo - 50): hi + 50] = True
    n_bg = 0
    edges = np.flatnonzero(np.diff(np.concatenate(([1], busy.astype(np.int8), [1]))))
    for g0, g1 in zip(edges[::2], edges[1::2]):
        for c0 in range(g0, g1 - 150 + 1, 300):
            c1 = min(c0 + 300, g1)
            piece = x[c0 * HOP: c1 * HOP]
            w = words_of(asr(piece))
            if any(t in OKAY or re.match(r"^(gl|cl|kl)", t) for t in w):
                rows.append((path.name, c0 / 100, c1 / 100, "skip: background had okay/gl word", " ".join(w)))
                continue
            write(out / "background" / f"{tag}_bg{n_bg:04d}.wav", piece)
            rows.append((path.name, c0 / 100, c1 / 100, "background negative", " ".join(w)))
            n_bg += 1

    counts = {"positive": 0, "negative": 0}
    for lo, hi, decision, text, clip in sorted(res, key=lambda r: r[0]):
        if clip is not None:
            kind = "positive" if decision.startswith("positive") else "negative"
            write(out / kind / f"{tag}_{counts[kind]:04d}.wav", clip)
            counts[kind] += 1
        rows.append((path.name, lo / 100, hi / 100, decision, text))
    print(f"  {path.name}: {counts['positive']} wake words, "
          f"{counts['negative']} speech negatives, {n_bg} background negatives", flush=True)
    return rows


# --------------------------------------------------------------------------- #
# hard-negative sessions
# --------------------------------------------------------------------------- #


def suspicious_times(words):
    """Times of anything that might be the wake word."""
    toks = [words_of(w["w"]) for w in words]
    hits = []
    for i, t in enumerate(toks):
        t = t[0] if t else ""
        nxt = toks[i + 1][0] if i + 1 < len(toks) and toks[i + 1] else ""
        if re.match(r"^(glad|glados|gladys|claros|lados)", t) or (
                t in OKAY and re.match(r"^(gl|cl|kl)", nxt)):
            hits.append(words[i]["t0"])
    return hits


def segment_negative_session(asr, path: Path, out: Path, tag: str, cache: Path):
    x = load16k(path)
    key = hashlib.sha1(x[:: 97].tobytes()).hexdigest()[:12]
    cached = cache / f"{path.stem}-{key}.json"
    if cached.exists():
        words = json.loads(cached.read_text())
    else:
        print(f"  transcribing {path.name} ({len(x) / SR / 60:.0f} min) ...", flush=True)
        words = asr.words_with_times(x)
        cache.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(words))
    bad = suspicious_times(words)

    rows, n_train, n_test, n_dropped, test_s = [], 0, 0, 0, 0.0
    block = int(NEG_BLOCK_S * SR)
    for bi, start in enumerate(range(0, len(x), block)):
        audio = x[start: start + block]
        t0 = start / SR
        if len(audio) < SR * NEG_CHUNK_S:
            continue
        if bi % HOLDOUT_EVERY == HOLDOUT_EVERY - 1:
            write(out / "home_test" / f"{tag}_block{bi:03d}.wav", audio)
            n_test += 1
            test_s += len(audio) / SR
            rows.append((path.name, t0, t0 + len(audio) / SR, "home test block", ""))
            continue
        step = int(NEG_CHUNK_S * SR)
        for ci in range(0, len(audio) - step + 1, step):
            c0 = t0 + ci / SR
            if any(c0 - EXCLUSION_S <= t <= c0 + NEG_CHUNK_S + EXCLUSION_S for t in bad):
                n_dropped += 1
                rows.append((path.name, c0, c0 + NEG_CHUNK_S, "skip: near wake-word-like word", ""))
                continue
            write(out / "home_train" / f"{tag}_b{bi:03d}_{ci // step:04d}.wav",
                  audio[ci: ci + step])
            n_train += 1
    print(f"  {path.name}: {n_train} training chunks, {n_test} held-out blocks "
          f"({test_s / 60:.1f} min), {n_dropped} chunks dropped near "
          f"{len(bad)} wake-word-like words", flush=True)
    return rows


# --------------------------------------------------------------------------- #


def duration_s(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return float(out.stdout.strip() or 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="whisper.cpp ggml model file")
    args = ap.parse_args()

    root = config.RECORDINGS_DIR
    files = sorted(p for p in root.rglob("*") if p.is_file()
                   and p.suffix.lower() in AUDIO_SUFFIXES)
    neg_files = [p for p in files if "hard_negatives" in p.relative_to(root).parts]
    pos_sessions = [p for p in files if p not in neg_files
                    and duration_s(p) > config.MAX_SINGLE_TAKE_S]
    if not pos_sessions and not neg_files:
        print("No long recording sessions found -- nothing to segment.")
        return

    out = config.SEGMENTED_DIR
    cache = out / ".transcripts"
    for sub in ("positive", "negative", "background", "home_train", "home_test"):
        shutil.rmtree(out / sub, ignore_errors=True)
    asr = Transcriber(args.model)

    def tag_of(p):
        rel = p.relative_to(root).with_suffix("")
        return re.sub(r"[^A-Za-z0-9]+", "_", str(rel)).strip("_")

    rows = []
    print(f"Positive sessions ({len(pos_sessions)})")
    for p in pos_sessions:
        rows += segment_positive_session(asr, p, out, tag_of(p))
    print(f"Hard-negative sessions ({len(neg_files)})")
    for p in neg_files:
        rows += segment_negative_session(asr, p, out, tag_of(p), cache)

    manifest = out / "manifest.tsv"
    manifest.write_text("file\tstart_s\tend_s\tdecision\ttranscript\n" + "".join(
        f"{f}\t{a:.2f}\t{b:.2f}\t{d}\t{t}\n" for f, a, b, d, t in rows))
    print("\nTotals:")
    for sub in ("positive", "negative", "background", "home_train", "home_test"):
        print(f"  {sub:11s} {len(list((out / sub).glob('*.wav'))):5d} clips")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
