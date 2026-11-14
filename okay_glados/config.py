"""Central configuration for the "Okay GLaDOS" microWakeWord training pipeline.

Every stage of the pipeline imports this module so that paths, sample counts and
hyperparameters live in exactly one place.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = PROJECT_ROOT / "okay_glados"
WORK_DIR = PIPELINE_DIR / "work"

# Where *you* drop hand-recorded wav files of the wake word.  See
# okay_glados/recordings/README.md.
RECORDINGS_DIR = PIPELINE_DIR / "recordings"

# Long recording *sessions* (many takes in one file, or an hour of room audio
# in recordings/hard_negatives/) are split into individual clips by
# tools/segment_recordings.py.  Its output lives here:
#   positive/         one wake word per clip
#   negative/         other speech pulled from the positive sessions
#   home_train/       3 s chunks of the hard-negative room audio (trained on)
#   home_test/        5 min blocks of that room audio, never trained on
SEGMENTED_DIR = WORK_DIR / "recordings_segmented"
# A file in recordings/ longer than this is a session, not a single take.
MAX_SINGLE_TAKE_S = 4.0

DOWNLOAD_DIR = WORK_DIR / "downloads"
PIPER_DIR = WORK_DIR / "piper-sample-generator"

# Augmentation sources
RIR_DIR = WORK_DIR / "mit_rirs"
BACKGROUND_AUDIOSET_DIR = WORK_DIR / "audioset_16k"
BACKGROUND_FMA_DIR = WORK_DIR / "fma_16k"

# Held-out real-world audio used only as an extra ambient *test* set.
AMBIENT_EXTRA_DIR = WORK_DIR / "audioset_ambient_16k"

# TTS output
POSITIVE_TRAIN_DIR = WORK_DIR / "positive_train"
POSITIVE_HELDOUT_DIR = WORK_DIR / "positive_heldout"
ADVERSARIAL_DIR = WORK_DIR / "adversarial"

# Spectrogram feature output (mmap ninja)
FEATURES_DIR = WORK_DIR / "features"
FEATURES_POSITIVE = FEATURES_DIR / "positive"
# Held-out positives rendered with *mild* augmentation only -- someone speaking
# normally to the device in an ordinary room.  Evaluation-only; never trained
# on, and deliberately not listed in the training config.
FEATURES_POSITIVE_CLEAN = FEATURES_DIR / "positive_clean"
FEATURES_POSITIVE_RECORDED = FEATURES_DIR / "positive_recorded"
FEATURES_ADVERSARIAL = FEATURES_DIR / "adversarial"
FEATURES_AMBIENT_EXTRA = FEATURES_DIR / "ambient_extra"
# Held-out blocks of your own room audio, streamed for false accepts per hour.
FEATURES_HOME_AMBIENT = FEATURES_DIR / "home_ambient"

# Pre-generated negative spectrograms downloaded from Hugging Face
NEGATIVE_DIR = WORK_DIR / "negative_datasets"

# Training output
TRAIN_DIR = WORK_DIR / "trained_models" / "okay_glados"
REPORT_DIR = PIPELINE_DIR / "report"

# The virtualenv that holds torch + piper (kept separate from the training env
# because piper-sample-generator pins numpy<2 while microWakeWord needs numpy>=2)
PIPER_PYTHON = PROJECT_ROOT / ".venv-piper" / "bin" / "python"
TRAIN_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"

# --------------------------------------------------------------------------- #
# The wake word
# --------------------------------------------------------------------------- #

WAKE_WORD = "Okay GLaDOS"

# IMPORTANT --------------------------------------------------------------- #
# Piper/eSpeak mispronounces the literal string "Okay GLaDOS" as
# "okay gee-la-doss"  (oʊkˈeɪ dʒˈiː lˌæ dˈɑːs) because it reads the mixed-case
# acronym letter by letter.  "Okay glados" is also wrong -- it produces
# "okay GLAH-dose" (oʊkˈeɪ ɡlˈɑːdoʊz).
#
# The phoneticised spelling "Okay gladohs" produces the intended
#     oʊkˈeɪ ɡlˈædoʊz     ("oh-KAY GLAD-ohz")
# and is therefore the primary spelling used for generation.  Never put the
# literal "Okay glados" / "Okay GLaDOS" spelling in this list.
# -------------------------------------------------------------------------- #

# (spelling, share of the positive sample budget).  Shares must sum to 1.0.
# All of these phonemise to a correct-sounding GLaDOS; the primary
# phoneticisation dominates, the rest add pronunciation variety so the model
# generalises to how different people actually say it.
POSITIVE_SPELLINGS = [
    ("Okay gladohs", 0.59),    # oʊkˈeɪ ɡlˈædoʊz   <- the required phoneticisation
    ("Okay, gladohs.", 0.14),  # same, with a comma pause + falling intonation
    ("Okay gladohce", 0.14),   # oʊkˈeɪ ɡlˈædoʊs   voiceless final /s/
    ("Ok gladohs", 0.13),      # ˌoʊkˈeɪ ɡlˈædoʊz  weaker stress on "o-"
]

# Every spelling above must phonemise with this nucleus.  "ɡlˈɑːd" ("GLAH-dos")
# is what the rejected "Okay glados" spelling produces, and spellings such as
# "Okay glahdohs" phonemise to exactly the same thing -- they are excluded on
# purpose.  tools/check_pronunciation.py enforces this.
REQUIRED_PHONEMES = "\u0261l\u02c8\u00e6d"  # ɡlˈæd
FORBIDDEN_PHONEMES = "\u0261l\u02c8\u0251\u02d0d"  # ɡlˈɑːd

# Phrases that sound similar to the wake word but must NOT trigger it.  These
# are generated with the same TTS and trained as hard negatives; they are what
# keeps the false-accept rate down on near-miss speech.
ADVERSARIAL_PHRASES = [
    "gladohs",
    "hey gladohs",
    "okay",
    "okay glasses",
    "okay glass",
    "okay class",
    "okay ladders",
    "okay gladiator",
    "okay gladiolus",
    "glad ohs",
    "glad hose",
    "the lady goes",
    "okay motos",
    "okay nabu",
    "okay computer",
    "okay google",
    "hey google",
    "alexa",
    "okay potatoes",
    "okay tomatoes",
    "a glass of soda",
    "okay so",
    "okay dose",
    "clay dose",
    "play those",
    "okay ghost",
    "okay toast",
    "let's go",
    "okay let those",
    "we had a lot of those",
    "okay gladly",
    "okay lattice",
    "day glow",
    "okay bravo",
    "gladus",
    "glados is a robot",
    "okay stop",
    "okay next",
]

# --------------------------------------------------------------------------- #
# Sample generation budget
# --------------------------------------------------------------------------- #

# Speakers 0..TRAIN_MAX_SPEAKER-1 of the LibriTTS-R model are used for training
# samples; speakers TRAIN_MAX_SPEAKER..HELDOUT_MAX_SPEAKER-1 are reserved for
# the validation/test sets.  Disjoint speakers keep the benchmark honest.
TRAIN_MAX_SPEAKER = 700
HELDOUT_MIN_SPEAKER = 700
HELDOUT_MAX_SPEAKER = 900

N_POSITIVE_TRAIN = 30000
N_POSITIVE_HELDOUT = 2500
N_ADVERSARIAL_TRAIN = 12000
N_ADVERSARIAL_HELDOUT = 1200

# --------------------------------------------------------------------------- #
# Feature generation
# --------------------------------------------------------------------------- #

WINDOW_STEP_MS = 10          # matches the v2 ESPHome models (feature_step_size)
CLIP_DURATION_MS = 1500      # longest wake word the streaming model accepts
AUGMENTATION_DURATION_S = 3.2
STRIDE = 3                   # model's first-layer stride: one inference / 30 ms

# Frames the model consumes per (non-streaming) inference, derived the same way
# microwakeword.model_train_eval.load_config derives it:
#   final layer  = 1 + (16000*1.5 - 480) // (STRIDE * 160)              = 50
#   dropped      = (first_conv_kernel - 1)
#                  + sum(max(mixconv kernel) - 1) * STRIDE              = 154
# Any spectrogram shorter than this cannot be fed to the model, so
# un-augmented feature sets are filtered against it.
SPECTROGRAM_LENGTH = 204

# --------------------------------------------------------------------------- #
# Production-quality acceptance thresholds
# --------------------------------------------------------------------------- #
# Measured on the *streaming, quantized* TFLite model against the held-out
# positive samples (false rejects) and the DiPCo dinner-party corpus plus a
# held-out AudioSet slice (false accepts).
# The shipped operating point is the most sensitive cutoff whose measured false
# accept rate is at or under TARGET_FAPH -- about one spurious wake per ten
# hours of continuous audio.  One per hour (microWakeWord's own curves run out
# to two) would be ~24 a day, which nobody would live with; on the other hand
# demanding *zero* observed accepts over a finite test set makes the operating
# point hostage to two or three individual audio events, so it is reported
# alongside rather than used as the target.
TARGET_FAPH = 0.1
TARGET_FRR = 0.05            # <= 5% false rejects on normal-use speech there
