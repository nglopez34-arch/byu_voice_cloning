from faster_whisper import WhisperModel, BatchedInferencePipeline
from pathlib import Path
from tqdm import tqdm
import torch

# Auto-detect which author's clips to transcribe
print("Looking for clip directories...")
clip_dirs = list(Path(".").glob("*_clips"))
if not clip_dirs:
    print("❌ No clip directories found!")
    print("   Run segment_audio.py first to create clips.")
    exit(1)

if len(clip_dirs) > 1:
    print("\nMultiple clip directories found:")
    for i, dir in enumerate(clip_dirs, 1):
        wav_count = len(list(dir.rglob("*.wav")))
        speech_count = len([d for d in dir.iterdir() if d.is_dir()])
        print(f"  {i}. {dir.name} ({speech_count} speeches, {wav_count} clips)")

    choice = int(input("\nWhich one do you want to transcribe? Enter number: ")) - 1
    clips_base = clip_dirs[choice]
else:
    clips_base = clip_dirs[0]

author_slug = clips_base.name.replace('_clips', '')
print(f"\n✓ Processing: {author_slug.replace('_', '-')}")

# Configuration
model_size = "large-v3"
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

print(f"Loading faster-whisper {model_size} on {device} ({compute_type})...")
model = WhisperModel(model_size, device=device, compute_type=compute_type)
pipeline = BatchedInferencePipeline(model=model)

# Batch size — how many segments decoded in parallel on GPU
# Tune based on your VRAM (RTX 5080 should handle 16-24 easily)
BATCH_SIZE = 16

speech_dirs = [d for d in clips_base.iterdir() if d.is_dir()]
print(f"Found {len(speech_dirs)} speeches to transcribe\n")

total_clips = 0
transcribed = 0

for speech_dir in speech_dirs:
    print(f"\n{'=' * 60}")
    print(f"Processing: {speech_dir.name}")
    print(f"{'=' * 60}")

    clips = sorted(speech_dir.glob("*.wav"))
    if not clips:
        print("  No clips found, skipping")
        continue

    print(f"  Found {len(clips)} clips")
    transcripts = []

    for clip in tqdm(clips, desc="  Transcribing"):
        total_clips += 1

        # Batched pipeline handles internal batching of audio segments
        segments, info = pipeline.transcribe(
            str(clip),
            language="en",
            task="transcribe",
            batch_size=BATCH_SIZE,
        )

        transcript = " ".join(seg.text.strip() for seg in segments)
        transcripts.append(transcript)

        txt_file = clip.with_suffix('.txt')
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(transcript)

        transcribed += 1

#If, for some reason, you want transcriptions of the full speeches you can uncomment this section.
#     full_transcript = " ".join(transcripts)
#     combined_file = speech_dir / f"{speech_dir.name}_FULL.txt"
#     with open(combined_file, 'w', encoding='utf-8') as f:
#         f.write(full_transcript)

    print(f"  ✓ Transcribed {len(clips)} clips")
    print(f"  ✓ Combined transcript: {combined_file.name}")

print(f"\n{'=' * 60}")
print(f"✅ Transcription Complete!")
print(f"   Total clips transcribed: {transcribed}")
print(f"   Each clip has individual .txt file")
print(f"{'=' * 60}")