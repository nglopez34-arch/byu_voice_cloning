import torch
import torchaudio
from pathlib import Path
from pydub import AudioSegment
import numpy as np
from multiprocessing import Pool, cpu_count
import functools

torch.set_num_threads(1)

# VAD parameters for natural sentence breaks
MIN_SILENCE_DURATION = 800
MIN_SPEECH_DURATION = 5000
MAX_SPEECH_DURATION = 30000
SPEECH_PAD = 300


def process_speech(mp3_file, speech_idx, total_files, output_base):
    """Process a single speech file - runs in its own process."""
    # Each process loads its own VAD model (can't share across processes)
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    print(f"[{speech_idx}/{total_files}] Processing: {mp3_file.name}")

    speech_name = mp3_file.stem
    clips_dir = output_base / speech_name
    clips_dir.mkdir(exist_ok=True)

    # Load audio with pydub
    audio = AudioSegment.from_mp3(mp3_file)

    # Convert to mono 16kHz WAV for VAD
    audio_16k = audio.set_channels(1).set_frame_rate(16000)

    # Use a unique temp file per process to avoid collisions
    temp_wav = f"temp_for_vad_{speech_idx}.wav"
    audio_16k.export(temp_wav, format="wav")

    wav = read_audio(temp_wav, sampling_rate=16000)

    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=16000,
        threshold=0.5,
        min_speech_duration_ms=MIN_SPEECH_DURATION,
        max_speech_duration_s=MAX_SPEECH_DURATION / 1000,
        min_silence_duration_ms=MIN_SILENCE_DURATION,
        window_size_samples=512,
        speech_pad_ms=SPEECH_PAD
    )

    print(f"  [{speech_name}] Found {len(speech_timestamps)} speech segments")

    clip_count = 0
    for i, timestamp in enumerate(speech_timestamps, 1):
        start_sample_16k = timestamp['start']
        end_sample_16k = timestamp['end']

        start_ms = int((start_sample_16k / 16000) * 1000)
        end_ms = int((end_sample_16k / 16000) * 1000)

        segment = audio[start_ms:end_ms]
        duration_sec = len(segment) / 1000

        if duration_sec < 3:
            continue

        if duration_sec > 20:
            mid = len(segment) // 2
            segments_to_save = [segment[:mid], segment[mid:]]
        else:
            segments_to_save = [segment]

        for seg in segments_to_save:
            clip_count += 1
            output_path = clips_dir / f"{speech_name}_clip_{clip_count:04d}.wav"
            seg.export(
                output_path,
                format="wav",
                parameters=["-ar", "22050", "-ac", "1"]
            )

    # Cleanup temp file
    Path(temp_wav).unlink()

    print(f"  ✓ [{speech_name}] Created {clip_count} clips")
    return clip_count


if __name__ == "__main__":
    # Auto-detect which author's speeches to process
    print("Looking for speech directories...")
    speech_dirs = list(Path(".").glob("*_speeches"))

    if not speech_dirs:
        print("❌ No speech directories found!")
        print("   Run fetch_audio.py first to download speeches.")
        exit(1)

    if len(speech_dirs) > 1:
        print("\nMultiple speech directories found:")
        for i, d in enumerate(speech_dirs, 1):
            mp3_count = len(list(d.glob("*.mp3")))
            print(f"  {i}. {d.name} ({mp3_count} MP3 files)")
        choice = int(input("\nWhich one do you want to segment? Enter number: ")) - 1
        speeches_dir = speech_dirs[choice]
    else:
        speeches_dir = speech_dirs[0]

    author_slug = speeches_dir.name.replace('_speeches', '')
    print(f"\n✓ Processing: {author_slug.replace('_', '-')}")

    output_base = Path(f"{author_slug}_clips")
    output_base.mkdir(exist_ok=True)

    mp3_files = sorted(speeches_dir.glob("*.mp3"))

    if not mp3_files:
        print(f"❌ No MP3 files found in {speeches_dir}")
        exit(1)

    num_workers = min(cpu_count(), len(mp3_files))
    total_files = len(mp3_files)

    print(f"\n{'=' * 60}")
    print(f"Speech Segmentation Settings:")
    print(f"  Min silence for split: {MIN_SILENCE_DURATION}ms")
    print(f"  Min clip length: {MIN_SPEECH_DURATION / 1000}s")
    print(f"  Max clip length: {MAX_SPEECH_DURATION / 1000}s")
    print(f"  Parallel workers: {num_workers}")
    print(f"  Files to process: {total_files}")
    print(f"{'=' * 60}\n")

    # Build args for starmap
    args = [
        (mp3_file, idx, total_files, output_base)
        for idx, mp3_file in enumerate(mp3_files, 1)
    ]

    with Pool(processes=num_workers) as pool:
        clip_counts = pool.starmap(process_speech, args)

    total_clips = sum(clip_counts)

    print(f"\n{'=' * 60}")
    print(f"✅ Segmentation Complete!")
    print(f"   Total clips created: {total_clips}")
    print(f"   Output directory: {output_base.absolute()}")
    print(f"\n   Next: Run transcription script on clips")
    print(f"{'=' * 60}")