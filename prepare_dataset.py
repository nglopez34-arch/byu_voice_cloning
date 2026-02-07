"""
prepare_dataset.py

Scans the clips directory, validates all audio/text pairs,
resamples audio to 22050 Hz mono WAV, filters by duration, and produces
the metadata files that XTTS v2 fine-tuning expects.

LJSpeech format required by the trainer:
    metadata.csv uses pipe '|' delimiter with columns: file_id|text|text
    Audio files go in:  dataset/wavs/{file_id}.wav

Outputs:
    ~/PycharmProjects/tts_model/dataset_{author}/
        wavs/                  - Processed WAV files (22050 Hz, mono)
        metadata_train.csv     - Training metadata
        metadata_eval.csv      - Evaluation metadata
"""

import os
import random
from pathlib import Path

import soundfile as sf
import librosa
import numpy as np
from tqdm import tqdm

# ── Auto-detect clips directory ────────────────────────────────────────────────
print("Looking for clip directories...")
clip_dirs = list(Path(".").glob("*_clips"))

if not clip_dirs:
    print("❌ No clip directories found!")
    print("   Run segment_audio.py and transcribe_clips.py first.")
    exit(1)

if len(clip_dirs) > 1:
    print("\nMultiple clip directories found:")
    for i, dir in enumerate(clip_dirs, 1):
        wav_count = len(list(dir.rglob("*.wav")))
        txt_count = len(list(dir.rglob("*.txt")))
        speech_count = len([d for d in dir.iterdir() if d.is_dir()])
        print(f"  {i}. {dir.name} ({speech_count} speeches, {wav_count} clips, {txt_count} transcripts)")

    choice = int(input("\nWhich one do you want to prepare? Enter number: ")) - 1
    CLIPS_DIR = clip_dirs[choice]
else:
    CLIPS_DIR = clip_dirs[0]

# Extract author slug from directory name
author_slug = CLIPS_DIR.name.replace('_clips', '')
print(f"\n✓ Processing: {author_slug.replace('_', '-')}")

# ── Configuration ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path.home() / "PycharmProjects" / "tts_model"
OUTPUT_DIR = PROJECT_ROOT / f"dataset_{author_slug}"  # Author-specific dataset
WAVS_DIR = OUTPUT_DIR / "wavs"

TARGET_SR = 22050
MIN_DURATION = 2.0    # seconds – skip very short clips
MAX_DURATION = 20.0   # seconds – skip overly long clips (XTTS max_wav_length ~11.6s)
EVAL_SPLIT = 0.05     # 5% of clips held out for evaluation
SEED = 42
# ───────────────────────────────────────────────────────────────────────────────


def find_clip_pairs(clips_dir: Path) -> list[dict]:
    """Walk every speech folder and pair each .wav with its .txt transcript."""
    pairs = []
    for speech_folder in sorted(clips_dir.iterdir()):
        if not speech_folder.is_dir():
            continue
        wav_files = sorted(speech_folder.glob("*.wav"))
        for wav_path in wav_files:
            txt_path = wav_path.with_suffix(".txt")
            if not txt_path.exists():
                # Try case-insensitive match
                candidates = list(speech_folder.glob(wav_path.stem + ".*"))
                txt_candidates = [c for c in candidates if c.suffix.lower() == ".txt"]
                if txt_candidates:
                    txt_path = txt_candidates[0]
                else:
                    print(f"  ⚠  No transcript for {wav_path.name}, skipping")
                    continue
            pairs.append({
                "wav": wav_path,
                "txt": txt_path,
                "speech": speech_folder.name,
            })
    return pairs


def process_audio(wav_path: Path, out_path: Path) -> float | None:
    """Load, resample to 22050 Hz mono, save. Returns duration or None if bad."""
    try:
        audio, sr = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
    except Exception as e:
        print(f"  ⚠  Could not load {wav_path.name}: {e}")
        return None

    duration = len(audio) / TARGET_SR
    if duration < MIN_DURATION or duration > MAX_DURATION:
        return None

    # Normalize peak to -1 dBFS
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9

    sf.write(str(out_path), audio, TARGET_SR, subtype="PCM_16")
    return duration


def read_transcript(txt_path: Path) -> str:
    """Read and clean a transcript file."""
    text = txt_path.read_text(encoding="utf-8", errors="replace").strip()
    # Collapse whitespace
    text = " ".join(text.split())
    return text


def main():
    random.seed(SEED)

    print(f"\nScanning clips in: {CLIPS_DIR}")
    pairs = find_clip_pairs(CLIPS_DIR)
    print(f"Found {len(pairs)} wav/txt pairs across {len(set(p['speech'] for p in pairs))} speeches\n")

    if not pairs:
        print("ERROR: No clip pairs found. Check your directory structure.")
        return

    # Prepare output dirs
    WAVS_DIR.mkdir(parents=True, exist_ok=True)

    accepted = []
    total_duration = 0.0

    print("Processing audio files...")
    for pair in tqdm(pairs, desc="Processing"):
        # Create a unique file_id (no extension, no path — just the stem)
        file_id = f"{pair['speech']}__{pair['wav'].stem}"
        out_path = WAVS_DIR / f"{file_id}.wav"

        duration = process_audio(pair["wav"], out_path)
        if duration is None:
            continue

        transcript = read_transcript(pair["txt"])
        if len(transcript) < 2:
            print(f"  ⚠  Empty/short transcript for {pair['wav'].name}, skipping")
            continue

        accepted.append({
            "file_id": file_id,
            "text": transcript,
            "duration": duration,
        })
        total_duration += duration

    print(f"\nAccepted {len(accepted)} clips, total duration: {total_duration/60:.1f} minutes")

    # Shuffle and split
    random.shuffle(accepted)
    n_eval = max(1, int(len(accepted) * EVAL_SPLIT))
    eval_set = accepted[:n_eval]
    train_set = accepted[n_eval:]

    # Write metadata CSVs in LJSpeech format:  file_id|text|text
    # The ljspeech formatter reads cols[0] as the file ID and cols[2] as the text.
    # cols[1] is the "original text" (we just duplicate the text).
    for split_name, split_data in [("metadata_train.csv", train_set), ("metadata_eval.csv", eval_set)]:
        csv_path = OUTPUT_DIR / split_name
        with open(csv_path, "w", encoding="utf-8") as f:
            for row in split_data:
                # LJSpeech format: file_id|text|text  (pipe-separated, no header)
                # Escape any pipes in the text itself
                text = row["text"].replace("|", " ")
                f.write(f"{row['file_id']}|{text}|{text}\n")
        print(f"Wrote {csv_path}  ({len(split_data)} entries)")

    # Summary
    train_dur = sum(r["duration"] for r in train_set)
    eval_dur = sum(r["duration"] for r in eval_set)
    print(f"\n{'='*60}")
    print(f"  Train: {len(train_set)} clips, {train_dur/60:.1f} min")
    print(f"  Eval:  {len(eval_set)} clips, {eval_dur/60:.1f} min")
    print(f"  Total: {total_duration/60:.1f} min ({total_duration/3600:.2f} hrs)")
    print(f"{'='*60}")
    print(f"\nDataset ready at: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()