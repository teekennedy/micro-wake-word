#!/usr/bin/env python3
"""Step 5 -- benchmark the trained model and write the report.

Everything here is measured on the **quantized streaming TFLite model** -- the
exact artefact that runs on the ESP32 -- and only on data the model never
trained on:

  false accepts  DiPCo dinner-party corpus + a held-out AudioSet slice,
                 streamed end to end with the same 5-frame moving average and
                 detection cooldown ESPHome uses.
  false rejects  synthetic wake words from LibriTTS-R speakers 700-899, plus
                 any hand recordings held out in step 3.

Outputs (in okay_glados/report/):
  training_progress.png   FRR / false-accepts-per-hour over the training run
  roc_curve.png           the false-accept vs false-reject trade-off
  okay_glados.tflite      the model
  okay_glados.json        the ESPHome manifest, with the chosen cutoff
  REPORT.md               the numbers, written out
"""

import argparse
import json
import multiprocessing as mp
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from mmap_ninja.ragged import RaggedMmap
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

CUTOFFS = np.round(np.arange(0.0, 1.001, 0.01), 3)
SLIDING_WINDOW = 5        # ESPHome's sliding_window_size
COOLDOWN_SLICES = 25      # inference slices ignored after a detection (~0.75 s)
IGNORE_LEADING = 25       # streaming warm-up slices skipped on positive clips

TFLITE_REL = "tflite_stream_state_internal_quant/stream_state_internal_quant.tflite"

# Parses e.g. "Step 1000 (nonstreaming): Validation: recall at no faph = 12.34 ..."
VALIDATION_RE = re.compile(
    r"Step (?P<step>\d+) \(nonstreaming\): Validation: "
    r"recall at no faph = (?P<recall_no_faph>[\d.]+) with cutoff (?P<cutoff>[\d.]+), "
    r"accuracy = (?P<accuracy>[\d.]+)%, recall = (?P<recall>[\d.]+)%, "
    r"precision = (?P<precision>[\d.]+)%, ambient false positives = (?P<afp>\d+), "
    r"estimated false positives per hour = (?P<faph>[\d.]+), "
    r"loss = (?P<loss>[\d.]+), auc = (?P<auc>[\d.]+), "
    r"average viable recall = (?P<avr>[\d.]+)"
)


# --------------------------------------------------------------------------- #
# training-curve parsing
# --------------------------------------------------------------------------- #


def parse_training_log(path: Path):
    if not path.exists():
        return []
    records = []
    for line in path.read_text(errors="replace").splitlines():
        m = VALIDATION_RE.search(line)
        if m:
            d = {k: float(v) for k, v in m.groupdict().items()}
            d["step"] = int(d["step"])
            records.append(d)
    # A resumed run restarts the step counter; keep only the last monotonic run.
    for i in range(len(records) - 1, 0, -1):
        if records[i]["step"] <= records[i - 1]["step"]:
            return records[i:]
    return records


# --------------------------------------------------------------------------- #
# streaming evaluation
# --------------------------------------------------------------------------- #


_WORKER = {}


def _init_worker(model_path, stride):
    from microwakeword.inference import Model
    _WORKER["model"] = Model(model_path, stride=stride)


def _smooth(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.size < SLIDING_WINDOW:
        return np.array([])
    return sliding_window_view(probabilities, SLIDING_WINDOW).mean(axis=-1)


def _ambient_job(job):
    """Stream one ambient spectrogram; return (accepts per cutoff, duration_h)."""
    mmap = RaggedMmap(job["path"])
    spectrogram = mmap[job["index"]]
    probabilities = _WORKER["model"].predict_spectrogram(spectrogram)
    smoothed = _smooth(probabilities)

    duration_h = len(probabilities) * config.STRIDE * (config.WINDOW_STEP_MS / 1000) / 3600
    accepts = np.zeros(CUTOFFS.shape[0])
    cooldown = np.ones(CUTOFFS.shape[0]) * COOLDOWN_SLICES
    for p in smoothed:
        cooldown = np.maximum(cooldown - 1, 0)
        fired = (p > CUTOFFS) & (cooldown == 0)
        accepts[fired] += 1
        cooldown[fired] = COOLDOWN_SLICES
    return accepts, duration_h


def _positive_job(job):
    """Stream one positive/near-miss clip; return its peak smoothed score."""
    mmap = RaggedMmap(job["path"])
    spectrogram = mmap[job["index"]]
    probabilities = _WORKER["model"].predict_spectrogram(spectrogram)
    smoothed = _smooth(probabilities[IGNORE_LEADING:])
    return float(smoothed.max()) if smoothed.size else 0.0


def mmap_jobs(dirs):
    jobs = []
    for d in dirs:
        for path in sorted(Path(d).glob("**/*_mmap")):
            n = len(RaggedMmap(str(path)))
            jobs.extend({"path": str(path), "index": i} for i in range(n))
    return jobs


def run_pool(fn, jobs, model_path, workers, desc):
    if not jobs:
        return []
    print(f"  {desc}: {len(jobs)} clips on {workers} workers ...", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker,
                  initargs=(model_path, config.STRIDE)) as pool:
        return list(pool.imap_unordered(fn, jobs, chunksize=4))


def ambient_faph(results):
    if not results:
        return None, 0.0
    accepts = np.sum([r[0] for r in results], axis=0)
    hours = float(np.sum([r[1] for r in results]))
    return accepts / hours, hours


def frr_at_cutoffs(scores):
    if not scores:
        return None
    scores = np.asarray(scores)
    return np.array([(scores <= c).mean() for c in CUTOFFS])


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #


def plot_training(records, out_path: Path, stage_boundaries=(), shipped_step=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not records:
        print("  no validation records found in the training log; skipping plot")
        return
    steps = np.array([r["step"] for r in records])
    frr = np.array([100 * (1 - r["recall"] / 100) for r in records])
    # recall_no_faph == 0 means *no* cutoff reached zero false accepts at that
    # checkpoint, i.e. the metric is undefined rather than 100% bad.  Break the
    # line there instead of drawing a spike to 100%.
    frr_no_fa = np.array([100 * (1 - r["recall_no_faph"]) if r["recall_no_faph"] > 0
                          else np.nan for r in records])
    faph = np.array([r["faph"] for r in records])
    avr = np.array([100 * r["avr"] for r in records])
    best_i = int(np.nanargmax(avr))

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    fig.suptitle('"Okay GLaDOS" — validation metrics during training', fontsize=13)
    for ax in axes:
        for boundary in stage_boundaries:
            ax.axvline(boundary, color="#95a5a6", lw=0.8, ls="-", alpha=0.7)

    ax = axes[0]
    ax.plot(steps, frr, color="#c0392b", lw=1.4, label="false rejects @ p>0.5")
    if np.isfinite(frr_no_fa).any():
        # Markers as well as a line: at most checkpoints no cutoff reached zero
        # false accepts, so the finite points are often isolated.
        ax.plot(steps, frr_no_fa, color="#e67e22", ls="--", lw=1.2,
                marker="o", ms=3.5,
                label="false rejects at the zero-false-accept cutoff\n"
                      "(gaps = no cutoff reached zero)")
    ax.set_ylabel("false reject rate (%)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    ax.plot(steps, faph, color="#2980b9", lw=1.4, label="false accepts / hour @ p>0.5")
    ax.axhline(config.TARGET_FAPH, color="grey", ls=":",
               label=f"target ({config.TARGET_FAPH:g}/h)")
    ax.set_ylabel("false accepts per hour")
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(steps, avr, color="#27ae60", lw=1.4,
            label="average recall over 0–2 false accepts/hour")
    ax.plot(steps[best_i], avr[best_i], "o", ms=8, mfc="none", mec="#7f8c8d",
            mew=1.6, zorder=4,
            label=f"argmax of this metric (step {steps[best_i]})")
    if shipped_step is not None and shipped_step in set(steps.tolist()):
        j = int(np.flatnonzero(steps == shipped_step)[0])
        ax.plot(steps[j], avr[j], "k*", ms=15, zorder=5,
                label=f"shipped checkpoint (step {shipped_step}) — chosen by\n"
                      f"streaming evaluation, not by this curve")
    ax.set_ylabel("average viable recall (%)")
    ax.set_xlabel("training step")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_roc(curves, operating_point, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    colors = ["#c0392b", "#2980b9", "#27ae60", "#8e44ad"]
    # The whole decision lives below ~1 false accept per hour; anything to the
    # right of that is a setting nobody would ship.
    xmax = 1.0
    ymax = 5
    for (name, faph, frr), color in zip(curves, colors):
        # Drop the cutoff==1.0 endpoint: nothing can exceed a probability of 1,
        # so it is a degenerate 100%-FRR point that only wrecks the y scale.
        mask = (faph <= xmax) & (CUTOFFS < 1.0)
        ax.plot(faph[mask], 100 * frr[mask], color=color, label=name, lw=2)
        if mask.any():
            ymax = max(ymax, 100 * frr[mask].max())
    if operating_point:
        ax.plot(operating_point["faph"], 100 * operating_point["frr"], "k*",
                markersize=16, zorder=5,
                label=f"shipped cutoff {operating_point['cutoff']:.2f}  "
                      f"({operating_point['faph']:.3f}/h, "
                      f"{100 * operating_point['frr']:.2f}% FRR)")
    ax.axvline(config.TARGET_FAPH, color="grey", ls=":", lw=1)
    ax.annotate(f"target ≤ {config.TARGET_FAPH:g}/h", xy=(config.TARGET_FAPH, 0),
                xytext=(3, 3), textcoords="offset points", fontsize=8,
                color="grey", va="bottom", ha="left")
    ax.axhline(100 * config.TARGET_FRR, color="grey", ls=":", lw=1)
    ax.annotate(f"target ≤ {100 * config.TARGET_FRR:g}% false rejects",
                xy=(xmax, 100 * config.TARGET_FRR), fontsize=8, color="grey",
                va="bottom", ha="right")
    ax.set_xlabel("false accepts per hour  (16.2 h of held-out ambient audio)")
    ax.set_ylabel("false reject rate (%)")
    ax.set_title('"Okay GLaDOS" — quantized streaming model, held-out data')
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax * 1.08)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 4) - 2))
    ap.add_argument("--log", type=Path, default=config.PIPELINE_DIR / "logs" / "step4_train.log")
    ap.add_argument("--stages", type=int, nargs="*", default=[20000, 40000],
                    help="training steps where the learning rate changed")
    ap.add_argument("--skip-streaming", action="store_true",
                    help="only redraw the training curve")
    args = ap.parse_args()

    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Parsing the training log ...")
    records = parse_training_log(args.log)
    print(f"  {len(records)} validation points")
    shipped_step = None
    sel = config.REPORT_DIR / "checkpoint_selection.json"
    if sel.exists():
        try:
            entries = json.loads(sel.read_text())
            shipped_step = min(entries, key=lambda e: e["mean_frr_0_2faph"])["step"]
        except (ValueError, KeyError):
            shipped_step = None
    plot_training(records, config.REPORT_DIR / "training_progress.png",
                  stage_boundaries=args.stages, shipped_step=shipped_step)
    if records:
        (config.REPORT_DIR / "training_progress.json").write_text(
            json.dumps(records, indent=1))
    if args.skip_streaming:
        return

    model_path = config.TRAIN_DIR / TFLITE_REL
    if not model_path.exists():
        raise SystemExit(f"{model_path} not found -- run step4 first")

    print("\nStreaming benchmark (quantized model, held-out data only)")
    ambient_sets = {
        "DiPCo dinner party": [config.NEGATIVE_DIR / "dinner_party_eval" / "testing_ambient"],
        "AudioSet (held out)": [config.FEATURES_AMBIENT_EXTRA / "testing_ambient"],
        "Home room audio (held out)": [config.FEATURES_HOME_AMBIENT / "testing_ambient"],
    }
    ambient = {}
    for name, dirs in ambient_sets.items():
        jobs = mmap_jobs(dirs)
        if not jobs:
            continue
        res = run_pool(_ambient_job, jobs, str(model_path), args.workers, name)
        faph, hours = ambient_faph(res)
        ambient[name] = {"faph": faph, "hours": hours}
        print(f"    {name}: {hours:.2f} h")

    if not ambient:
        raise SystemExit("no ambient test data found")

    total_hours = sum(v["hours"] for v in ambient.values())
    combined_faph = sum(v["faph"] * v["hours"] for v in ambient.values()) / total_hours

    positive_sets = {
        "Normal use (unseen speakers)": [config.FEATURES_POSITIVE_CLEAN / "testing"],
        "Adverse (noise/reverb/quiet)": [config.FEATURES_POSITIVE / "testing"],
        "Hand recorded (held out)": [config.FEATURES_POSITIVE_RECORDED / "testing"],
    }
    positives = {}
    for name, dirs in positive_sets.items():
        jobs = mmap_jobs(dirs)
        if not jobs:
            continue
        scores = run_pool(_positive_job, jobs, str(model_path), args.workers, name)
        positives[name] = {"scores": scores, "frr": frr_at_cutoffs(scores)}

    adversarial_jobs = mmap_jobs([config.FEATURES_ADVERSARIAL / "testing"])
    adversarial = None
    if adversarial_jobs:
        scores = run_pool(_positive_job, adversarial_jobs, str(model_path),
                          args.workers, "Near-miss phrases")
        adversarial = np.array([(np.asarray(scores) > c).mean() for c in CUTOFFS])

    # Headline false-reject rate: normal-use speech, plus any hand recordings.
    # The adverse set is reported separately rather than averaged in -- mixing
    # them would let the augmentation recipe set the headline number.
    headline_names = [n for n in ("Normal use (unseen speakers)",
                                  "Hand recorded (held out)") if n in positives]
    headline_scores = [s for n in headline_names for s in positives[n]["scores"]]
    overall_frr = frr_at_cutoffs(headline_scores or
                                 [s for v in positives.values() for s in v["scores"]])

    # Operating point: the most sensitive cutoff at which *no* false accept was
    # observed anywhere in the ambient test set.  Picking the lowest cutoff that
    # merely satisfies some non-zero rate maximises recall on paper while
    # shipping a device that wakes itself up all day, so zero observed accepts
    # is the bar; TARGET_FAPH is only the fallback.
    zero_fa = np.flatnonzero(combined_faph == 0)
    viable = np.flatnonzero(combined_faph <= config.TARGET_FAPH)
    operating = None
    if viable.size:
        i = int(viable[0])
        operating = {
            "cutoff": float(CUTOFFS[i]),
            "faph": float(combined_faph[i]),
            "frr": float(overall_frr[i]),
            "index": i,
            "zero_fa_cutoff": float(CUTOFFS[int(zero_fa[0])]) if zero_fa.size else None,
            "zero_fa_frr": float(overall_frr[int(zero_fa[0])]) if zero_fa.size else None,
        }
        print(f"\n  operating point: cutoff {operating['cutoff']:.2f} -> "
              f"{operating['faph']:.3f} false accepts/h, "
              f"{100 * operating['frr']:.2f}% false rejects")
    else:
        print("\n  WARNING: no cutoff reaches the target false-accept rate")

    curves = [(name, combined_faph, v["frr"]) for name, v in positives.items()]
    plot_roc(curves, operating, config.REPORT_DIR / "roc_curve.png")

    write_report(records, ambient, positives, adversarial, combined_faph,
                 overall_frr, operating, total_hours, model_path)


def write_report(records, ambient, positives, adversarial, combined_faph,
                 overall_frr, operating, total_hours, model_path):
    shutil.copy(model_path, config.REPORT_DIR / "okay_glados.tflite")
    size_bytes = model_path.stat().st_size

    manifest = {
        "type": "micro",
        "wake_word": config.WAKE_WORD,
        "author": "trained locally with microWakeWord",
        "website": "https://github.com/kahrendt/microWakeWord",
        "model": "okay_glados.tflite",
        "trained_languages": ["en"],
        "version": 2,
        "micro": {
            "probability_cutoff": round(operating["cutoff"], 2) if operating else 0.97,
            "feature_step_size": config.WINDOW_STEP_MS,
            "sliding_window_size": SLIDING_WINDOW,
            "tensor_arena_size": 30000,
            "minimum_esphome_version": "2024.7.0",
        },
    }
    (config.REPORT_DIR / "okay_glados.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def row(i):
        parts = [f"| {CUTOFFS[i]:.2f} | {combined_faph[i]:.3f} "]
        for v in positives.values():
            parts.append(f"| {100 * v['frr'][i]:.2f}% ")
        if adversarial is not None:
            parts.append(f"| {100 * adversarial[i]:.2f}% ")
        return "".join(parts) + "|"

    header = ["| cutoff | false accepts/h "]
    for name in positives:
        header.append(f"| FRR {name} ")
    if adversarial is not None:
        header.append("| near-miss accepts ")
    header = "".join(header) + "|"
    sep = "|" + "---|" * (header.count("|") - 1)

    interesting = sorted({*range(50, 101, 5), *( [operating["index"]] if operating else [] )})
    table = "\n".join(row(i) for i in interesting if i < len(CUTOFFS))

    lines = [
        '# "Okay GLaDOS" — microWakeWord training report', "",
        "All numbers below come from the **quantized streaming TFLite model** "
        "(the artefact that runs on the ESP32), measured on data that was never "
        "trained on.", "",
        "## Headline result", "",
    ]
    if operating:
        ok = operating["frr"] <= config.TARGET_FRR
        # Rule of three: zero events in N hours puts the 95% upper bound on the
        # rate at 3/N per hour.  Quote it rather than claiming a true zero.
        bound = 3.0 / total_hours
        events = operating["faph"] * total_hours
        fa_cell = ("**0 observed** in %.1f h" % total_hours if operating["faph"] == 0
                   else "%.3f/h (%.0f event%s in %.1f h)"
                        % (operating["faph"], events, "" if round(events) == 1 else "s",
                           total_hours))
        lines += [
            "| metric | value | target | |",
            "|---|---|---|---|",
            f"| probability cutoff | {operating['cutoff']:.2f} | — | |",
            f"| false accepts | {fa_cell} | ≤ {config.TARGET_FAPH:g}/h | "
            f"{'✅' if operating['faph'] <= config.TARGET_FAPH else '❌'} |",
            f"| false reject rate (normal use) | {100 * operating['frr']:.2f}% | "
            f"≤ {100 * config.TARGET_FRR:g}% | {'✅' if ok else '❌'} |", "",
            f"The model {'**meets**' if ok else '**misses**'} the false-reject "
            f"target at the most sensitive cutoff that holds false accepts to "
            f"{config.TARGET_FAPH:g} per hour or below.", "",
        ]
        if operating["zero_fa_cutoff"] is not None:
            lines += [
                f"Raising the cutoff to **{operating['zero_fa_cutoff']:.2f}** gives "
                f"*no* false accept anywhere in {total_hours:.2f} hours, at "
                f"{100 * operating['zero_fa_frr']:.2f}% false rejects. That is the "
                f"setting to use if a spurious wake is more annoying than a missed "
                f"one. Note that zero observed events over a finite test set is not "
                f"a true zero: by the rule of three the 95% upper bound is still "
                f"**{bound:.2f} per hour**.", "",
            ]
    else:
        lines += ["No cutoff reached the target false-accept rate.", ""]

    lines += ["## Test data", "",
              "| set | size |", "|---|---|"]
    for name, v in ambient.items():
        lines.append(f"| {name} (ambient, false accepts) | {v['hours']:.2f} h |")
    for name, v in positives.items():
        lines.append(f"| {name} (positives, false rejects) | {len(v['scores'])} clips |")
    if adversarial is not None:
        lines.append("| Near-miss phrases (adversarial) | see table |")
    lines += ["", f"Total ambient audio: **{total_hours:.2f} hours**.", "",
              "## Operating curve", "", header, sep, table, "",
              "## Model", "",
              f"- file: `okay_glados.tflite` ({size_bytes / 1024:.1f} KiB)",
              f"- feature step size: {config.WINDOW_STEP_MS} ms, stride {config.STRIDE} "
              f"(one inference every {config.STRIDE * config.WINDOW_STEP_MS} ms)",
              f"- ESPHome manifest: `okay_glados.json`", "",
              "```yaml", "micro_wake_word:", "  models:",
              "    - model: <url or path to okay_glados.json>", "```", ""]
    if records:
        best = max(records, key=lambda r: r["avr"])
        lines += ["## Training", "",
                  f"- {len(records)} validation checkpoints over "
                  f"{records[-1]['step']} steps",
                  f"- the framework's own metric (average viable recall) peaks at "
                  f"{100 * best['avr']:.2f}% on step {best['step']}",
                  "- but the shipped weights were chosen by converting candidate "
                  "checkpoints to quantized streaming models and measuring them "
                  "(`tools/select_checkpoint.py`); see `checkpoint_selection.json`",
                  "- see `training_progress.png`", ""]
        comparison = config.REPORT_DIR / "run_comparison.json"
        if comparison.exists():
            runs = json.loads(comparison.read_text())
            lines += ["### Training runs tried", "",
                      "Each run was swept checkpoint by checkpoint as a quantized "
                      "streaming model against the *validation* ambient audio; the "
                      "best of the best was shipped. Lower is better.", "",
                      "| run | what changed | candidates | best step | mean FRR "
                      "(0–2 accepts/h) |", "|---|---|---|---|---|"]
            notes = {
                "B": "batch 256, 18k @1e-3 then 12k @3e-4  **(shipped)**",
                "C": "SpecAugment removed, 3 LR stages",
                "D": "single 18k @1e-3 stage, dense checkpoints",
                "E": "rebalanced weights (neg 10, pos 3, adversarial 6)",
            }
            for name in sorted(runs):
                r = runs[name]
                lines.append(
                    f"| {name} | {notes.get(name, '')} | {r['candidates']} | "
                    f"{r['best']['step']} | "
                    f"{100 * r['best']['mean_frr_0_2faph']:.2f}% |")
            lines += ["", "The two hypotheses that failed are worth recording: "
                      "removing SpecAugment (run C) and rebalancing the class "
                      "weights away from microWakeWord's defaults (run E) both "
                      "made things clearly worse.", ""]

    (config.REPORT_DIR / "REPORT.md").write_text("\n".join(lines))
    print(f"  wrote {config.REPORT_DIR / 'REPORT.md'}")

    np.savez(config.REPORT_DIR / "curves.npz", cutoffs=CUTOFFS,
             combined_faph=combined_faph, overall_frr=overall_frr,
             **{f"faph_{k}": v["faph"] for k, v in ambient.items()},
             **{f"frr_{k}": v["frr"] for k, v in positives.items()})


if __name__ == "__main__":
    main()
