"""
generate_audio.py

Generate speech from arbitrarily long text (multiple paragraphs).
Splits text into sentence-sized chunks, generates audio for each,
concatenates into a single WAV, saves it, and plays it on your Ubuntu desktop.

The script will prompt you for the text you want to convert to speech.
"""

from pathlib import Path
import re
import os
import subprocess
import shutil
import time
from datetime import datetime

import torch
import torchaudio
import soundfile as sf
import numpy as np

# PyTorch 2.6+ monkey-patch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# ── Auto-detect training output directory ──────────────────────────────────────
PROJECT_ROOT = Path.home() / "PycharmProjects" / "tts_model"

print("Looking for trained models...")
training_dirs = list(PROJECT_ROOT.glob("training_output_*"))

if not training_dirs:
    print("❌ No trained models found!")
    print("   Run finetune_xtts.py first.")
    exit(1)

if len(training_dirs) > 1:
    print("\nMultiple trained models found:")
    for i, dir in enumerate(training_dirs, 1):
        author = dir.name.replace('training_output_', '').replace('_', '-')
        # Count checkpoints
        ckpt_count = len(list(dir.rglob("*.pth")))
        print(f"  {i}. {author} ({ckpt_count} checkpoint files)")

    choice = int(input("\nWhich model do you want to use? Enter number: ")) - 1
    TRAINING_OUTPUT_DIR = training_dirs[choice]
else:
    TRAINING_OUTPUT_DIR = training_dirs[0]

# Extract author slug from directory name
author_slug = TRAINING_OUTPUT_DIR.name.replace('training_output_', '')
print(f"\n✓ Using model: {author_slug.replace('_', '-')}")

DATASET_DIR = PROJECT_ROOT / f"dataset_{author_slug}"

# Output directory for generated audio
OUTPUT_DIR = Path("generated_audio")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────────
SAMPLE_RATE = 24000

# Generation parameters
LANGUAGE = "en"
TEMPERATURE = 0.65
TOP_K = 50
TOP_P = 0.85
SPEED = 1.0
REPETITION_PENALTY = 5.0

# Pause duration (in seconds) inserted between sentences/chunks
SENTENCE_PAUSE = 0.25
PARAGRAPH_PAUSE = 0.6
# ───────────────────────────────────────────────────────────────────────────────


def find_best_checkpoint(output_dir: Path):
    """
    Find the best model checkpoint based on validation loss.
    Prioritizes files named 'best_model*.pth' which are created by the trainer.
    """
    candidates = []

    # First priority: best_model files (these are based on validation metrics)
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            full = Path(root) / f
            if f.startswith("best_model") and f.endswith(".pth"):
                # Extract step number if present
                match = re.search(r'best_model_(\d+)', f)
                step = int(match.group(1)) if match else 0
                candidates.insert(0, (step, full))

    # If no best_model found, look for any checkpoint
    if not candidates:
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                full = Path(root) / f
                if f.endswith(".pth") and "checkpoint" in f.lower():
                    candidates.append((0, full))

    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {output_dir}")

    # Sort by step number (highest first) and take the best
    candidates.sort(reverse=True, key=lambda x: x[0])
    checkpoint = candidates[0][1]

    # Find config.json
    config_path = checkpoint.parent / "config.json"
    if not config_path.exists():
        for p in checkpoint.parent.rglob("config.json"):
            config_path = p
            break

    print(f"  Using checkpoint: {checkpoint.name}")
    return str(checkpoint), str(config_path)


def find_vocab_file() -> str:
    search_locations = [
        TRAINING_OUTPUT_DIR / "XTTS_v2.0_original_model_files" / "vocab.json",
        Path.home() / ".local" / "share" / "tts" / "tts_models--multilingual--multi-dataset--xtts_v2" / "vocab.json",
    ]
    for path in search_locations:
        if path.exists():
            return str(path)
    raise FileNotFoundError("vocab.json not found")


def find_speaker_file() -> str | None:
    search_locations = [
        Path.home() / ".local" / "share" / "tts" / "tts_models--multilingual--multi-dataset--xtts_v2" / "speakers_xtts.pth",
        TRAINING_OUTPUT_DIR / "XTTS_v2.0_original_model_files" / "speakers_xtts.pth",
    ]
    for path in search_locations:
        if path.exists():
            return str(path)
    return None


def find_reference_clip(dataset_dir: Path, min_dur: float = 6.0, max_dur: float = 15.0) -> str:
    """Auto-select a reference clip from the dataset based on duration."""
    wavs_dir = dataset_dir / "wavs"
    if not wavs_dir.exists():
        # Try clips directory as fallback
        clips_dir = Path(f"{author_slug}_clips")
        if clips_dir.exists():
            wavs_dir = clips_dir

    best = None
    best_score = float("inf")
    best_dur = 0.0

    for wav_path in wavs_dir.rglob("*.wav"):
        try:
            info = sf.info(str(wav_path))
            duration = info.duration
        except Exception:
            continue
        if min_dur <= duration <= max_dur:
            score = abs(duration - 10.0)
            if score < best_score:
                best_score = score
                best = wav_path
                best_dur = duration

    if best is None:
        # Fallback: just use first WAV
        wav_files = list(wavs_dir.rglob("*.wav"))
        if not wav_files:
            raise FileNotFoundError("No WAV files found for reference")
        best = wav_files[0]
        best_dur = sf.info(str(best)).duration

    print(f"  Auto-selected reference: {best.name} ({best_dur:.1f}s)")
    return str(best)


def get_or_create_reference_clip(checkpoint_dir: Path, dataset_dir: Path) -> str:
    """
    Get the reference clip for this model checkpoint.
    If one already exists in the checkpoint directory, use that.
    Otherwise, auto-select one from the dataset and save it to the checkpoint directory.
    """
    reference_clip_path = checkpoint_dir / "reference_clip.wav"

    if reference_clip_path.exists():
        # Use existing reference clip
        try:
            info = sf.info(str(reference_clip_path))
            duration = info.duration
            print(f"  Using saved reference: reference_clip.wav ({duration:.1f}s)")
            return str(reference_clip_path)
        except Exception as e:
            print(f"  Warning: Saved reference clip appears corrupted, selecting new one...")

    # No existing reference clip, select and save one
    print("  No saved reference clip found, selecting one...")
    selected_clip = find_reference_clip(dataset_dir)

    # Copy the selected clip to the checkpoint directory
    shutil.copy(selected_clip, reference_clip_path)
    print(f"  Saved reference clip to: {reference_clip_path}")

    return str(reference_clip_path)


def split_text_into_chunks(text: str, max_chars: int = 250):
    """
    Split long text into sentence-level chunks suitable for XTTS generation.
    Returns a list of (chunk_text, is_paragraph_end) tuples.
    """
    # Normalize whitespace but preserve paragraph breaks
    text = text.strip()
    paragraphs = re.split(r"\n\s*\n", text)

    chunks = []
    for p_idx, paragraph in enumerate(paragraphs):
        # Clean up the paragraph
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue

        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)

        # Group sentences into chunks that fit within max_chars
        current_chunk = ""
        for s_idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue

            if current_chunk and len(current_chunk) + len(sentence) + 1 > max_chars:
                is_para_end = False
                chunks.append((current_chunk, is_para_end))
                current_chunk = sentence
            else:
                current_chunk = f"{current_chunk} {sentence}".strip() if current_chunk else sentence

        # Flush remaining text in this paragraph
        if current_chunk:
            is_last_paragraph = (p_idx == len(paragraphs) - 1)
            is_para_end = not is_last_paragraph
            chunks.append((current_chunk, is_para_end))

    return chunks


def make_silence(duration_sec: float, sample_rate: int = 24000) -> np.ndarray:
    return np.zeros(int(duration_sec * sample_rate), dtype=np.float32)


def sanitize_filename(text: str, max_len: int = 50) -> str:
    """Create a safe filename from text."""
    # Take first portion of text
    text = text[:max_len]
    # Remove special characters
    text = re.sub(r'[^\w\s-]', '', text)
    # Replace spaces with underscores
    text = re.sub(r'[-\s]+', '_', text)
    # Remove leading/trailing underscores
    text = text.strip('_')
    return text if text else "generated_speech"


def get_text_input() -> str:
    """Prompt user for text input."""
    print("\n" + "=" * 60)
    print("TEXT INPUT")
    print("=" * 60)
    print("Choose an option:")
    print("  1. Type/paste text directly")
    print("  2. Load from a text file")

    choice = input("\nEnter choice (1 or 2): ").strip()

    if choice == "2":
        file_path = input("Enter path to text file: ").strip()
        file_path = Path(file_path.replace("'", "").replace('"', ''))

        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            exit(1)

        text = file_path.read_text(encoding='utf-8')
        print(f"✓ Loaded {len(text)} characters from {file_path.name}")
        return text

    else:
        print("\nEnter your text (press Ctrl+D on a new line when done):")
        print("-" * 60)
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass

        text = "\n".join(lines).strip()

        if not text:
            print("❌ No text provided!")
            exit(1)

        print(f"\n✓ Received {len(text)} characters")
        return text


def play_audio(filepath: str):
    """Play audio using the best available player on Ubuntu."""
    players = [
        # PulseAudio/PipeWire (most likely on modern Ubuntu desktop)
        ("paplay", ["paplay", filepath]),
        # ALSA direct
        ("aplay", ["aplay", "-q", filepath]),
        # SoX
        ("play", ["play", "-q", filepath]),
        # FFplay
        ("ffplay", ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", filepath]),
    ]

    for name, cmd in players:
        if shutil.which(name):
            print(f"  Playing with {name}...")
            try:
                subprocess.run(cmd, check=True)
                return
            except subprocess.CalledProcessError:
                continue

    print("  No audio player found. Install pulseaudio-utils, alsa-utils, sox, or ffmpeg.")
    print(f"  You can manually play: {filepath}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # ── Get text from user ─────────────────────────────────────────────────
    text = get_text_input()

    chunks = split_text_into_chunks(text)
    total_chars = sum(len(c) for c, _ in chunks)
    print(f"\nText split into {len(chunks)} chunks ({total_chars} characters)\n")

    # ── Load model ─────────────────────────────────────────────────────────
    print("Loading model...")
    ckpt_path, cfg_path = find_best_checkpoint(TRAINING_OUTPUT_DIR)
    vocab_path = find_vocab_file()
    speaker_file = find_speaker_file()

    config = XttsConfig()
    config.load_json(cfg_path)

    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_dir=str(Path(ckpt_path).parent),
        checkpoint_path=ckpt_path,
        vocab_path=vocab_path,
        speaker_file_path=speaker_file,
        eval=True,
        strict=False,
    )
    model = model.to(device)
    model.eval()

    if model.tokenizer is None:
        raise RuntimeError("Tokenizer failed to load — check vocab.json path")
    print("✓ Model loaded!\n")

    # ── Reference audio ────────────────────────────────────────────────────
    checkpoint_dir = Path(ckpt_path).parent
    ref_clip = get_or_create_reference_clip(checkpoint_dir, DATASET_DIR)

    print("Computing speaker latents...")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[ref_clip],
        gpt_cond_len=30,
        gpt_cond_chunk_len=6,
    )

    # ── Generate all chunks ────────────────────────────────────────────────
    print(f"\nGenerating speech...\n")
    all_audio = []
    start_time = time.time()

    for i, (chunk_text, is_paragraph_end) in enumerate(chunks):
        label = chunk_text[:70] + ("..." if len(chunk_text) > 70 else "")
        print(f"  [{i + 1}/{len(chunks)}] \"{label}\"")

        result = model.inference(
            text=chunk_text,
            language=LANGUAGE,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            speed=SPEED,
            repetition_penalty=REPETITION_PENALTY,
            enable_text_splitting=True,
        )

        all_audio.append(np.array(result["wav"], dtype=np.float32))

        # Insert pauses between chunks
        if i < len(chunks) - 1:
            pause = PARAGRAPH_PAUSE if is_paragraph_end else SENTENCE_PAUSE
            all_audio.append(make_silence(pause, SAMPLE_RATE))

    elapsed = time.time() - start_time

    # ── Concatenate and save ───────────────────────────────────────────────
    full_audio = np.concatenate(all_audio)
    duration = len(full_audio) / SAMPLE_RATE

    # Create descriptive filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    text_preview = sanitize_filename(text, max_len=40)
    filename = f"{timestamp}_{text_preview}.wav"
    output_path = OUTPUT_DIR / filename

    wav_tensor = torch.tensor(full_audio).unsqueeze(0)
    torchaudio.save(str(output_path), wav_tensor, SAMPLE_RATE)

    print(f"\n{'=' * 60}")
    print(f"✅ GENERATION COMPLETE!")
    print(f"{'=' * 60}")
    print(f"  Saved:      {output_path}")
    print(f"  Duration:   {duration:.1f}s")
    print(f"  Generated in {elapsed:.1f}s ({duration / elapsed:.1f}x realtime)")
    print(f"{'=' * 60}\n")

    # ── Play it ────────────────────────────────────────────────────────────
    #todo: uncomment below
    #play_audio(str(output_path))


if __name__ == "__main__":
    main()