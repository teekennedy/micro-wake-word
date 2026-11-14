#!/usr/bin/env python3
"""Step 4 -- write the training config and run microWakeWord's trainer.

Writes ``training_parameters.yaml`` describing every feature set and the
training schedule, then shells out to ``microwakeword.model_train_eval``.  The
trainer's stdout is teed to okay_glados/logs/step4_train.log, which step 5
parses to plot how the false-reject and false-accept rates evolve.

The model architecture is the same MixConv streaming network the production
ESPHome v2 models use: a 32-filter strided input convolution followed by four
mixed-depthwise blocks, one inference every 30 ms.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

_SUFFIX = f"_{config.RUN_NAME}" if config.RUN_NAME else ""
LOG_PATH = config.PIPELINE_DIR / "logs" / f"step4_train{_SUFFIX}.log"

# --- architecture (matches the v2 okay_nabu / hey_jarvis topology) ---------- #
MODEL_FLAGS = [
    "mixednet",
    "--pointwise_filters", "64,64,64,64",
    "--repeat_in_block", "1,1,1,1",
    "--mixconv_kernel_sizes", "[5], [7,11], [9,15], [23]",
    "--residual_connection", "0,0,0,0",
    "--first_conv_filters", "32",
    "--first_conv_kernel_size", "5",
    "--stride", str(config.STRIDE),
]


def has_features(features_dir: Path, split: str) -> bool:
    return features_dir.exists() and any((features_dir / split).glob("**/*_mmap"))


def any_split(features_dir: Path) -> bool:
    return features_dir.exists() and any(features_dir.glob("**/*_mmap"))


def build_config(training_steps, learning_rates, negative_class_weight,
                 eval_step_interval, batch_size, spec_augment_stages=0,
                 positive_weight=2.0, adversarial_weight=4.0, recorded_weight=0.75):
    n = len(training_steps)
    if spec_augment_stages:
        # Enable it only for the last `spec_augment_stages` stages.
        sizes = [0] * (n - spec_augment_stages) + [5] * spec_augment_stages
        counts = [0] * (n - spec_augment_stages) + [2] * spec_augment_stages
    else:
        sizes, counts = [0] * n, [0] * n
    spec_augment = (sizes, counts)

    features = []

    def add(features_dir, truth, sampling_weight, truncation_strategy,
            penalty_weight=1.0):
        if not any_split(Path(features_dir)):
            print(f"  (skipping absent feature set {features_dir})")
            return
        features.append({
            "features_dir": str(features_dir),
            "sampling_weight": sampling_weight,
            "penalty_weight": penalty_weight,
            "truth": truth,
            "truncation_strategy": truncation_strategy,
            "type": "mmap",
        })

    # --- positives ---------------------------------------------------- #
    add(config.FEATURES_POSITIVE, True, positive_weight, "truncate_start")
    # Hand-recorded wake word.  Weighted lower than the synthetic set because
    # it is far smaller: enough to pull the decision boundary toward the real
    # speaker, not enough to overfit a handful of clips.
    add(config.FEATURES_POSITIVE_RECORDED, True, recorded_weight, "truncate_start")

    # --- negatives ------------------------------------------------------ #
    # Near-miss phrases carry the most weight per sample: they are what keeps
    # "okay glasses" and "gladiolus" from firing the model.
    add(config.FEATURES_ADVERSARIAL, False, adversarial_weight, "truncate_start")
    add(config.NEGATIVE_DIR / "speech", False, 10.0, "random")
    add(config.NEGATIVE_DIR / "dinner_party", False, 10.0, "random")
    add(config.NEGATIVE_DIR / "no_speech", False, 5.0, "random")

    # --- evaluation-only sets ------------------------------------------- #
    # CHiME-6 dev+eval as validation_ambient (false accepts per hour during
    # training) and DiPCo as testing_ambient (final streaming benchmark).
    add(config.NEGATIVE_DIR / "dinner_party_eval", False, 0.0, "split")
    add(config.FEATURES_AMBIENT_EXTRA, False, 0.0, "split")
    # validation_ambient only: lets the trainer's own metric see music and TV.
    add(config.FEATURES_AMBIENT_VAL, False, 0.0, "split")
    add(config.FEATURES_HOME_AMBIENT, False, 0.0, "split")

    cfg = {
        "window_step_ms": config.WINDOW_STEP_MS,
        "train_dir": str(config.TRAIN_DIR),
        "features": features,
        "training_steps": training_steps,
        "positive_class_weight": [1] * len(training_steps),
        "negative_class_weight": negative_class_weight,
        "learning_rates": learning_rates,
        "batch_size": batch_size,
        # SpecAugment, off by default.  Measured on this task it made things
        # worse: in both runs where it was enabled for the later stages, the
        # streaming false-reject rate rose rather than fell.  The audio-level
        # augmentation in step 3 is already doing this job.
        "time_mask_max_size": spec_augment[0],
        "time_mask_count": spec_augment[1],
        "freq_mask_max_size": spec_augment[0],
        "freq_mask_count": spec_augment[1],
        "eval_step_interval": eval_step_interval,
        "clip_duration_ms": config.CLIP_DURATION_MS,
        # Best-weight selection: maximise the average recall achievable at
        # 0-2 false accepts per hour.  That single number is the thing a user
        # actually feels, so it is what we optimise rather than raw accuracy.
        "target_minimization": 0.9,
        "minimization_metric": None,
        "maximization_metric": "average_viable_recall",
    }
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="+", default=[15000, 15000, 10000])
    ap.add_argument("--learning-rates", type=float, nargs="+",
                    default=[0.001, 0.0005, 0.0001])
    ap.add_argument("--negative-class-weight", type=float, nargs="+", default=[20, 20, 20])
    ap.add_argument("--eval-step-interval", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--positive-weight", type=float, default=2.0,
                    help="sampling weight of the synthetic wake-word set")
    ap.add_argument("--adversarial-weight", type=float, default=4.0,
                    help="sampling weight of the near-miss phrase set")
    ap.add_argument("--recorded-weight", type=float, default=0.75,
                    help="sampling weight of the hand-recorded wake-word set")
    ap.add_argument("--spec-augment-stages", type=int, default=0,
                    help="enable SpecAugment for the last N stages (default 0: "
                         "off -- it measured worse on this task)")
    ap.add_argument("--restore", type=int, default=1,
                    help="resume from the checkpoint in train_dir if present")
    ap.add_argument("--fresh", action="store_true",
                    help="delete train_dir first and start from scratch")
    ap.add_argument("--train", type=int, default=1,
                    help="0 = skip training, just convert + benchmark best weights")
    ap.add_argument("--config-out", type=Path,
                    default=config.WORK_DIR / f"training_parameters{_SUFFIX}.yaml")
    ap.add_argument("--log", type=Path, default=LOG_PATH)
    args = ap.parse_args()

    n = len(args.steps)
    for name, lst in (("learning-rates", args.learning_rates),
                      ("negative-class-weight", args.negative_class_weight)):
        if len(lst) != n:
            raise SystemExit(f"--{name} must have {n} entries to match --steps")

    if args.fresh:
        shutil.rmtree(config.TRAIN_DIR, ignore_errors=True)

    cfg = build_config(args.steps, args.learning_rates, args.negative_class_weight,
                       args.eval_step_interval, args.batch_size,
                       args.spec_augment_stages, args.positive_weight,
                       args.adversarial_weight, args.recorded_weight)
    if not any(f["truth"] for f in cfg["features"]):
        raise SystemExit("No positive feature sets found -- run step3 first")

    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.config_out, "w") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False)
    print(f"Wrote {args.config_out}")
    for f in cfg["features"]:
        print(f"  {Path(f['features_dir']).name:22s} truth={str(f['truth']):5s} "
              f"weight={f['sampling_weight']:<5} {f['truncation_strategy']}")
    print(f"  total steps: {sum(args.steps)}  batch: {args.batch_size}  "
          f"eval every {args.eval_step_interval}")

    cmd = [
        str(config.TRAIN_PYTHON), "-u", "-m", "microwakeword.model_train_eval",
        f"--training_config={args.config_out}",
        "--train", str(args.train),
        "--restore_checkpoint", str(args.restore),
        "--test_tf_nonstreaming", "0",
        "--test_tflite_nonstreaming", "0",
        "--test_tflite_nonstreaming_quantized", "0",
        "--test_tflite_streaming", "0",
        "--test_tflite_streaming_quantized", "1",
        "--use_weights", "best_weights",
        *MODEL_FLAGS,
    ]
    args.log.parent.mkdir(parents=True, exist_ok=True)
    print("\n+ " + " ".join(cmd) + f"\n  (tee -> {args.log})\n", flush=True)

    # The trainer redraws a per-minibatch progress line with \r; keep only the
    # last redraw of each line so the log stays readable and small.
    with open(args.log, "ab") as logfile:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, cwd=config.PROJECT_ROOT)
        for raw in proc.stdout:
            line = raw.rsplit(b"\r", 1)[-1]
            if not line.strip():
                continue
            logfile.write(line if line.endswith(b"\n") else line + b"\n")
            logfile.flush()
            sys.stdout.buffer.write(line)
            sys.stdout.flush()
        rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"training failed with exit code {rc}")
    print("\nStep 4 complete.")


if __name__ == "__main__":
    main()
