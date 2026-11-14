#!/usr/bin/env python3
"""Pick the checkpoint that is actually best *as a quantized streaming model*.

microWakeWord keeps the weights that maximise ``average_viable_recall`` on the
non-streaming validation pass.  That metric is noisy from checkpoint to
checkpoint, and it is not the thing that ships: the artefact on the device is an
int8 streaming model whose scores are smoothed over 5 frames.  Taking the argmax
of a noisy proxy over 60 checkpoints reliably picks a lucky one.

So: take the top-N checkpoints by validation score, convert each one exactly as
the release build does, stream it over the **validation** ambient audio and
validation positives, and keep the one with the best real false-accept /
false-reject trade-off.  Selection uses validation data only -- the test sets
stay untouched for the final report.

    .venv/bin/python okay_glados/tools/select_checkpoint.py --candidates 10

Writes the winner to ``best_weights.weights.h5`` in the training directory so
that ``step4_train.py --train 0`` converts and benchmarks it.
"""

import argparse
import json
import multiprocessing as mp
import shutil
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import yaml
from mmap_ninja.ragged import RaggedMmap
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config as pipeline_config  # noqa: E402
from step5_evaluate import parse_training_log  # noqa: E402

CUTOFFS = np.round(np.arange(0.0, 1.001, 0.01), 3)
SLIDING_WINDOW = 5
COOLDOWN_SLICES = 25
IGNORE_LEADING = 25
CHUNK_FRAMES = 120_000  # ~20 min of audio per parallel job

_W = {}


def _init(model_path, stride):
    from microwakeword.inference import Model
    _W["model"] = Model(model_path, stride=stride)


def _smooth(p):
    p = np.asarray(p, dtype=np.float64)
    return sliding_window_view(p, SLIDING_WINDOW).mean(axis=-1) if p.size >= SLIDING_WINDOW \
        else np.array([])


def _ambient(job):
    return (job.get("set"), *_ambient_counts(job))


def _ambient_counts(job):
    spec = RaggedMmap(job["path"])[job["index"]][job["start"]:job["stop"]]
    probs = _W["model"].predict_spectrogram(spec)
    hours = len(probs) * pipeline_config.STRIDE * (pipeline_config.WINDOW_STEP_MS / 1000) / 3600
    accepts = np.zeros(CUTOFFS.shape[0])
    cooldown = np.ones(CUTOFFS.shape[0]) * COOLDOWN_SLICES
    for p in _smooth(probs):
        cooldown = np.maximum(cooldown - 1, 0)
        fired = (p > CUTOFFS) & (cooldown == 0)
        accepts[fired] += 1
        cooldown[fired] = COOLDOWN_SLICES
    return accepts, hours


def _positive(job):
    spec = RaggedMmap(job["path"])[job["index"]]
    s = _smooth(_W["model"].predict_spectrogram(spec)[IGNORE_LEADING:])
    return float(s.max()) if s.size else 0.0


def ambient_jobs(dirs):
    """Split long ambient spectrograms into chunks so they can be streamed in parallel.

    Resetting the streaming state at a chunk boundary costs a handful of warm-up
    slices out of ~40,000, which is immaterial for *ranking* checkpoints.  The
    final report in step 5 streams each clip whole.
    """
    jobs = []
    for d in dirs:
        for path in sorted(Path(d).glob("**/*_mmap")):
            m = RaggedMmap(str(path))
            for i in range(len(m)):
                n = m[i].shape[0]
                for start in range(0, n, CHUNK_FRAMES):
                    stop = min(start + CHUNK_FRAMES, n)
                    if stop - start >= pipeline_config.SPECTROGRAM_LENGTH:
                        jobs.append({"path": str(path), "index": i,
                                     "start": start, "stop": stop})
    return jobs


def positive_jobs(dirs, limit, seed=0):
    jobs = []
    for d in dirs:
        for path in sorted(Path(d).glob("**/*_mmap")):
            n = len(RaggedMmap(str(path)))
            jobs.extend({"path": str(path), "index": i} for i in range(n))
    if limit and len(jobs) > limit:
        rng = np.random.default_rng(seed)
        jobs = [jobs[i] for i in rng.choice(len(jobs), limit, replace=False)]
    return jobs


def convert_checkpoint(cfg, weights_path: Path, out_dir: Path, data_processor):
    """Reproduce the release conversion: streaming SavedModel -> int8 TFLite."""
    import microwakeword.mixednet as mixednet
    import microwakeword.utils as utils
    from microwakeword.layers import modes

    flags = Namespace(**cfg["flags"])
    model = mixednet.model(flags, shape=cfg["training_input_shape"], batch_size=1)
    model.load_weights(str(weights_path))

    local = dict(cfg)
    local["train_dir"] = str(out_dir)
    utils.convert_model_saved(model, local, folder="stream_state_internal",
                              mode=modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
    utils.convert_saved_model_to_tflite(
        local, audio_processor=data_processor,
        path_to_model=str(out_dir / "stream_state_internal"),
        folder=str(out_dir / "quant"), fname="model.tflite", quantize=True)
    return out_dir / "quant" / "model.tflite"


def evaluate(model_path, amb_jobs, pos_jobs, workers, rec_jobs=()):
    """Returns faph, synthetic FRR curve, hours, and the recorded FRR curve
    (None when there are no recorded validation clips)."""
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init,
                  initargs=(str(model_path), pipeline_config.STRIDE)) as pool:
        amb = list(pool.imap_unordered(_ambient, amb_jobs, chunksize=1))
        scores = list(pool.imap_unordered(_positive, pos_jobs, chunksize=8))
        rec = list(pool.imap_unordered(_positive, rec_jobs, chunksize=8))
    accepts = np.sum([a for _, a, _ in amb], axis=0)
    hours = float(np.sum([h for _, _, h in amb]))
    faph = accepts / hours
    by_set = {}
    for name in sorted({n for n, _, _ in amb if n}):
        h = float(np.sum([hh for n, _, hh in amb if n == name]))
        by_set[name] = (np.sum([a for n, a, _ in amb if n == name], axis=0) / h, h)

    def curve(s):
        s = np.asarray(s)
        return np.array([(s <= c).mean() for c in CUTOFFS])

    return faph, curve(scores), hours, (curve(rec) if rec else None), by_set


# The region the score integrates over.  It used to be 0-2 false accepts per
# hour (microWakeWord's own "viable" range), but the model ships at
# TARGET_FAPH = 0.1/h: a checkpoint could win on its behaviour at 1-2/h while
# being unusable where it is actually operated.
MAX_FAPH = 0.5


def summarise(faph, frr, max_faph=MAX_FAPH):
    """FRR at the shipping target, and the mean FRR across 0..max_faph accepts/hour."""
    viable = np.flatnonzero(faph <= pipeline_config.TARGET_FAPH)
    frr_at_target = float(frr[viable[0]]) if viable.size else 1.0
    order = np.argsort(faph, kind="stable")
    x, y = faph[order], frr[order]
    keep = x <= max_faph
    x, y = x[keep], y[keep]
    if x.size < 2:
        return frr_at_target, 1.0
    # Extend the curve to faph=0 and faph=max so every candidate integrates
    # over the same interval.
    xs = np.concatenate(([0.0], x, [max_faph]))
    ys = np.concatenate(([y[0]], y, [y[-1]]))
    mean_frr = float(np.trapezoid(ys, xs) / max_faph)
    return frr_at_target, mean_frr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=int, default=10)
    ap.add_argument("--select", choices=["avr", "uniform", "all"], default="avr",
                    help="'avr' takes the top-N by validation score; 'uniform' "
                         "spreads candidates across the whole run, which is "
                         "safer because the validation score is a weak proxy; "
                         "'all' sweeps every saved checkpoint and needs no log")
    ap.add_argument("--train-dir", type=Path, default=pipeline_config.TRAIN_DIR,
                    help="sweep the checkpoints of another (archived) run")
    ap.add_argument("--min-step", type=int, default=0)
    ap.add_argument("--positives", type=int, default=2000,
                    help="validation positives sampled per candidate")
    ap.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 4) - 2))
    ap.add_argument("--log", type=Path,
                    default=pipeline_config.LOG_DIR / "step4_train.log")
    ap.add_argument("--out", default="checkpoint_selection.json",
                    help="filename under report/ for the results (use a distinct "
                         "name per training run so sweeps do not overwrite "
                         "each other)")
    ap.add_argument("--apply", action="store_true",
                    help="overwrite best_weights.weights.h5 with the winner")
    args = ap.parse_args()

    train_dir = args.train_dir
    cfg = yaml.load((train_dir / "training_config.yaml").read_text(), yaml.Loader)

    if args.select == "all":
        steps = sorted(int(p.name.split("_")[2].split(".")[0])
                       for p in (train_dir / "train").glob("0_weights_*.weights.h5"))
        available = [{"step": s, "avr": float("nan")} for s in steps
                     if s >= args.min_step]
        chosen = available
    else:
        records = parse_training_log(args.log)
        if not records:
            raise SystemExit(f"no validation records in {args.log}")
        available = [r for r in records if r["step"] >= args.min_step and
                     (train_dir / "train" / f"0_weights_{r['step']}.weights.h5").exists()]
    if args.select == "all":
        pass
    elif args.select == "avr":
        chosen = sorted(available, key=lambda r: -r["avr"])[:args.candidates]
    else:
        idx = np.linspace(0, len(available) - 1, min(args.candidates, len(available)))
        chosen = [available[int(round(i))] for i in idx]
    seen, candidates = set(), []
    for r in sorted(chosen, key=lambda r: r["step"]):
        if r["step"] in seen:
            continue
        seen.add(r["step"])
        candidates.append((r["step"], r["avr"],
                           train_dir / "train" / f"0_weights_{r['step']}.weights.h5"))
    print(f"{len(candidates)} candidate checkpoints, steps "
          f"{candidates[0][0]}–{candidates[-1][0]}\n")

    import microwakeword.data as input_data
    data_processor = input_data.FeatureHandler(cfg)

    # Dinner-party speech (CHiME-6) *and* a held-out AudioSet slice: the first
    # has almost no music or TV, which is exactly what a wake word model in a
    # living room false-triggers on.
    amb_sets = {
        "chime": ambient_jobs([pipeline_config.NEGATIVE_DIR / "dinner_party_eval"
                               / "validation_ambient"]),
        "audioset": ambient_jobs([pipeline_config.FEATURES_AMBIENT_VAL
                                  / "validation_ambient"]),
        # Held-out room audio from the device itself, when there is any.
        "home": ambient_jobs([pipeline_config.FEATURES_HOME_AMBIENT
                              / "validation_ambient"]),
    }
    if not amb_sets["audioset"]:
        raise SystemExit("no AudioSet validation audio -- run step1 and "
                         "step3 --only ambient first")
    amb_jobs = [dict(j, set=name) for name, jobs in amb_sets.items() for j in jobs]
    pos_jobs = positive_jobs([pipeline_config.FEATURES_POSITIVE / "validation"],
                             args.positives)
    # Real recorded voices, scored separately and weighted equally with the
    # synthetic set -- otherwise ~2000 TTS clips would drown out a few dozen
    # real ones, and the real speakers are who the model is for.
    rec_jobs = positive_jobs([pipeline_config.FEATURES_POSITIVE_RECORDED / "validation"], 0)
    print(f"selection data: {len(amb_jobs)} ambient chunks, {len(pos_jobs)} positives, "
          f"{len(rec_jobs)} recorded positives\n")

    results, curves = [], {}
    for step, avr, ckpt in candidates:
        with tempfile.TemporaryDirectory(prefix="ckpt-") as tmp:
            tflite = convert_checkpoint(cfg, ckpt, Path(tmp), data_processor)
            faph, frr, hours, rec_frr, by_set = evaluate(
                tflite, amb_jobs, pos_jobs, args.workers, rec_jobs)
        frr_at_target, mean_frr = summarise(faph, frr)
        entry = {"step": step, "avr": avr, "hours": hours,
                 "frr_at_target": frr_at_target, "mean_frr": mean_frr,
                 "score": mean_frr}
        curves[f"faph_{step}"] = faph
        curves[f"frr_{step}"] = frr
        for name, (f, h) in by_set.items():
            curves[f"faph_{name}_{step}"] = f
            entry[f"hours_{name}"] = h
        extra = ""
        if rec_frr is not None:
            rec_at_target, rec_mean = summarise(faph, rec_frr)
            curves[f"rec_frr_{step}"] = rec_frr
            entry.update({"recorded_frr_at_target": rec_at_target,
                          "recorded_mean_frr": rec_mean,
                          "score": (mean_frr + rec_mean) / 2})
            extra = (f"  recorded {100 * rec_at_target:6.2f}% / "
                     f"{100 * rec_mean:6.2f}%")
        results.append(entry)
        print(f"  step {step:6d}  val_avr {avr:.4f}  synthetic FRR@target / mean 0-"
              f"{MAX_FAPH:g}/h {100 * frr_at_target:6.2f}% / {100 * mean_frr:6.2f}%"
              f"{extra}  score {100 * entry['score']:6.2f}%", flush=True)

    results.sort(key=lambda r: r["score"])
    best = results[0]
    print(f"\nBest streaming checkpoint: step {best['step']} "
          f"(score {100 * best['score']:.2f}%, synthetic FRR at "
          f"{pipeline_config.TARGET_FAPH:g}/h {100 * best['frr_at_target']:.2f}%)")
    out = pipeline_config.REPORT_DIR / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    np.savez(out.with_suffix(".npz"), cutoffs=CUTOFFS, **curves)
    print(f"wrote {out}")

    if args.apply:
        src = train_dir / "train" / f"0_weights_{best['step']}.weights.h5"
        dst = train_dir / "best_weights.weights.h5"
        shutil.copy(src, dst)
        print(f"copied {src.name} -> {dst.name}")
        print("now run: .venv/bin/python okay_glados/step4_train.py --train 0 ...")


if __name__ == "__main__":
    main()
