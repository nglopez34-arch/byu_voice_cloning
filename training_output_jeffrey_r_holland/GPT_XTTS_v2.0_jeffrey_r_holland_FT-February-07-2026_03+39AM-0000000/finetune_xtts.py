"""
finetune_xtts.py

Based directly on the official Coqui recipe:
    https://github.com/coqui-ai/TTS/blob/dev/recipes/ljspeech/xtts_v2/train_gpt_xtts.py

The GPT component of XTTS is fine-tuned while the DVAE and HiFi-GAN decoder
weights remain frozen. This script:
  1. Downloads dvae.pth and mel_stats.pth (required but not in the TTS model download)
  2. Configures GPTTrainerConfig to match the official recipe
  3. Uses load_tts_samples() to properly load the LJSpeech-formatted dataset
  4. Trains with the Trainer API

"""

import os
from pathlib import Path
from glob import glob

import torch

# PyTorch 2.6+ defaults weights_only=True, which Coqui TTS doesn't support yet.
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager

# ── Auto-detect dataset directory ──────────────────────────────────────────────
PROJECT_ROOT = Path.home() / "PycharmProjects" / "tts_model"

print("Looking for dataset directories...")
dataset_dirs = list(PROJECT_ROOT.glob("dataset_*"))

if not dataset_dirs:
    print("❌ No dataset directories found!")
    print("   Run prepare_dataset.py first.")
    exit(1)

if len(dataset_dirs) > 1:
    print("\nMultiple dataset directories found:")
    for i, dir in enumerate(dataset_dirs, 1):
        wav_count = len(list((dir / "wavs").glob("*.wav"))) if (dir / "wavs").exists() else 0
        train_csv = dir / "metadata_train.csv"
        train_lines = len(train_csv.read_text().splitlines()) if train_csv.exists() else 0
        author = dir.name.replace('dataset_', '').replace('_', '-')
        print(f"  {i}. {author} ({train_lines} training samples, {wav_count} WAV files)")

    choice = int(input("\nWhich dataset do you want to fine-tune on? Enter number: ")) - 1
    DATASET_DIR = dataset_dirs[choice]
else:
    DATASET_DIR = dataset_dirs[0]

# Extract author slug from directory name
author_slug = DATASET_DIR.name.replace('dataset_', '')
print(f"\n✓ Training on: {author_slug.replace('_', '-')}")

# ── Paths ──────────────────────────────────────────────────────────────────────
OUT_PATH = str(PROJECT_ROOT / f"training_output_{author_slug}")

# Where to store base model files (dvae, mel_stats, model, vocab)
CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files")

# ── Training hyperparameters ───────────────────────────────────────────────────
# Note from Coqui: BATCH_SIZE * GRAD_ACUMM_STEPS should ideally be >= 252
# On 16 GB VRAM we can't hit that, but fine-tuning still works well at smaller
# effective batch sizes.
BATCH_SIZE = 2
GRAD_ACUMM_STEPS = 4     # Handled in TrainerArgs, not GPTTrainerConfig
LEARNING_RATE = 5e-6
SAVE_STEP = 500
OPTIMIZER_WD_ONLY_ON_WEIGHTS = True

# ── Logging ────────────────────────────────────────────────────────────────────
RUN_NAME = f"GPT_XTTS_v2.0_{author_slug}_FT"
PROJECT_NAME = "XTTS_trainer"
DASHBOARD_LOGGER = "tensorboard"
LOGGER_URI = None

# ── HuggingFace download links for required training files ─────────────────────
# dvae.pth and mel_stats.pth are NOT included in the TTS model manager download.
# They must be fetched separately.
DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"
TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"
# ───────────────────────────────────────────────────────────────────────────────


def find_speaker_reference(dataset_dir: Path) -> str:
    """Find a good WAV from the dataset to use as speaker reference for test sentences."""
    import torchaudio
    wavs_dir = dataset_dir / "wavs"
    best = None
    best_score = float("inf")

    for wav_path in wavs_dir.glob("*.wav"):
        try:
            info = torchaudio.info(str(wav_path))
            dur = info.num_frames / info.sample_rate
        except Exception:
            continue
        if 6.0 <= dur <= 12.0:
            score = abs(dur - 8.0)
            if score < best_score:
                best_score = score
                best = str(wav_path)

    if best is None:
        # Fallback: just use the first WAV
        wavs = list(wavs_dir.glob("*.wav"))
        if wavs:
            best = str(wavs[0])
        else:
            raise FileNotFoundError(f"No WAV files found in {wavs_dir}")

    print(f"Speaker reference for test sentences: {Path(best).name}")
    return best


def main():
    # ── Verify dataset exists ──────────────────────────────────────────────
    train_csv = DATASET_DIR / "metadata_train.csv"
    eval_csv = DATASET_DIR / "metadata_eval.csv"
    wavs_dir = DATASET_DIR / "wavs"

    if not train_csv.exists():
        print(f"ERROR: {train_csv} not found. Run prepare_dataset.py first.")
        return

    if not wavs_dir.exists() or not any(wavs_dir.glob("*.wav")):
        print(f"ERROR: No WAV files in {wavs_dir}. Run prepare_dataset.py first.")
        return

    # ── Download / locate model files ──────────────────────────────────────
    os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

    DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, "dvae.pth")
    MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, "mel_stats.pth")
    TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, "vocab.json")
    XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, "model.pth")

    # Download dvae.pth and mel_stats.pth (required for training but not in the
    # standard TTS model download)
    if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
        print("Downloading DVAE and mel_stats files from HuggingFace...")
        ModelManager._download_model_files(
            [DVAE_CHECKPOINT_LINK, MEL_NORM_LINK],
            CHECKPOINTS_OUT_PATH,
            progress_bar=True,
        )

    # Download vocab.json and model.pth if needed
    if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
        print("Downloading XTTS v2.0 model files from HuggingFace...")
        ModelManager._download_model_files(
            [TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK],
            CHECKPOINTS_OUT_PATH,
            progress_bar=True,
        )

    # If model.pth wasn't downloaded (e.g. already exists via TTS model manager),
    # try to use the one from the TTS cache
    if not os.path.isfile(XTTS_CHECKPOINT):
        tts_cache_dir = Path.home() / ".local" / "share" / "tts" / "tts_models--multilingual--multi-dataset--xtts_v2"
        cached_model = tts_cache_dir / "model.pth"
        if cached_model.exists():
            print(f"Using cached model.pth from {cached_model}")
            XTTS_CHECKPOINT = str(cached_model)
        if not os.path.isfile(str(tts_cache_dir / "vocab.json")):
            pass
        else:
            TOKENIZER_FILE = str(tts_cache_dir / "vocab.json")

    # Verify all files exist
    for label, path in [
        ("DVAE checkpoint", DVAE_CHECKPOINT),
        ("Mel norm file", MEL_NORM_FILE),
        ("Tokenizer file", TOKENIZER_FILE),
        ("XTTS checkpoint", XTTS_CHECKPOINT),
    ]:
        if not os.path.isfile(path):
            print(f"ERROR: {label} not found at {path}")
            return
        size_mb = os.path.getsize(path) / 1e6
        print(f"  ✓ {label}: {path} ({size_mb:.1f} MB)")

    # ── Dataset config ─────────────────────────────────────────────────────
    config_dataset = BaseDatasetConfig(
        formatter="ljspeech",
        meta_file_train="metadata_train.csv",
        meta_file_val="metadata_eval.csv",
        path=str(DATASET_DIR),
        language="en",
    )

    # Find a speaker reference WAV for generating test sentences during training
    SPEAKER_REFERENCE = find_speaker_reference(DATASET_DIR)
    LANGUAGE = config_dataset.language

    # ── Model args ─────────────────────────────────────────────────────────
    model_args = GPTArgs(
        max_conditioning_length=132300,     # ~6 seconds at 22050 Hz
        min_conditioning_length=66150,      # ~3 seconds
        debug_loading_failures=False,
        max_wav_length=255995,              # ~11.6 seconds at 22050 Hz
        max_text_length=200,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,    # model to fine-tune
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )

    # ── Audio config ───────────────────────────────────────────────────────
    audio_config = XttsAudioConfig(
        sample_rate=22050,
        dvae_sample_rate=22050,
        output_sample_rate=24000,
    )

    # ── Training config ────────────────────────────────────────────────────
    # Parameters match the official Coqui recipe with adjustments for our
    # dataset size and GPU.
    config = GPTTrainerConfig(
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        run_description=f"GPT XTTS fine-tuning on {author_slug.replace('_', '-')} speech data",
        dashboard_logger=DASHBOARD_LOGGER,
        logger_uri=LOGGER_URI,
        audio=audio_config,
        batch_size=BATCH_SIZE,
        batch_group_size=48,
        eval_batch_size=BATCH_SIZE,
        num_loader_workers=4,
        eval_split_max_size=256,
        print_step=50,
        plot_step=100,
        log_model_step=1000,
        save_step=SAVE_STEP,
        save_n_checkpoints=3,
        save_checkpoints=True,
        print_eval=False,
        # Optimizer — matches official recipe (AdamW with tortoise-style params)
        optimizer="AdamW",
        optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=LEARNING_RATE,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={
            "milestones": [50000 * 18, 150000 * 18, 300000 * 18],
            "gamma": 0.5,
            "last_epoch": -1,
        },
        # Dataset
        datasets=[config_dataset],
        # Test sentences — generated at checkpoints to monitor voice quality
        test_sentences=[
            {
                "text": "My dear brothers and sisters, I am grateful to be with you today.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "Life is to be enjoyed, not just endured.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
        ],
    )

    # ── Build model ────────────────────────────────────────────────────────
    print("\nLoading pretrained XTTS v2 model via GPTTrainer...")
    model = GPTTrainer.init_from_config(config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # ── Load dataset ───────────────────────────────────────────────────────
    print("\nLoading dataset...")
    train_samples, eval_samples = load_tts_samples(
        [config_dataset],
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )
    print(f"  Train samples: {len(train_samples)}")
    print(f"  Eval samples:  {len(eval_samples)}")

    # ── Trainer ────────────────────────────────────────────────────────────
    trainer_args = TrainerArgs(
        restore_path=None,    # XTTS checkpoint is loaded via xtts_checkpoint key
        skip_train_epoch=False,
        start_with_eval=True,
        grad_accum_steps=GRAD_ACUMM_STEPS,
    )

    trainer = Trainer(
        trainer_args,
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    print(f"\nStarting fine-tuning...")
    print(f"  Batch size:       {BATCH_SIZE} (effective {BATCH_SIZE * GRAD_ACUMM_STEPS})")
    print(f"  Learning rate:    {LEARNING_RATE}")
    print(f"  Output dir:       {OUT_PATH}")
    print(f"  Save every:       {SAVE_STEP} steps")
    print()

    trainer.fit()

    print(f"\nTraining complete! Checkpoints saved in: {OUT_PATH}")
    print("Use inference.py to generate speech with your fine-tuned model.")


if __name__ == "__main__":
    main()