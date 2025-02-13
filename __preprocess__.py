import os
import cv2
import librosa
import soundfile as sf
import subprocess
import re

def preprocess_instrument_folders(data_dir, output_video_dir, output_audio_dir, segment_duration=10):
    """
    Preprocess all instrument folders, standardizing and segmenting videos and audio.

    Args:
        data_dir (str): Path to the directory containing instrument folders.
        output_video_dir (str): Directory to save processed video segments.
        output_audio_dir (str): Directory to save processed audio segments.
        segment_duration (int): Duration of each segment in seconds (default: 10 seconds).
    """
    os.makedirs(output_video_dir, exist_ok=True)
    os.makedirs(output_audio_dir, exist_ok=True)

    for instrument in os.listdir(data_dir):
        if instrument not in ["Cello", "DoubleBass", "Violin", "Viola"]:
            continue

        instrument_path = os.path.join(data_dir, instrument)
        if os.path.isdir(instrument_path):
            print(f"Processing instrument: {instrument}")
            instrument_video_dir = os.path.join(output_video_dir, instrument)
            instrument_audio_dir = os.path.join(output_audio_dir, instrument)
            os.makedirs(instrument_video_dir, exist_ok=True)
            os.makedirs(instrument_audio_dir, exist_ok=True)

            preprocess_folder(instrument_path, instrument_video_dir, instrument_audio_dir, segment_duration)

def preprocess_folder(input_dir, output_video_dir, output_audio_dir, segment_duration):
    """
    Preprocess videos and audio in a single instrument folder.

    Args:
        input_dir (str): Directory containing video and audio files.
        output_video_dir (str): Directory to save processed video segments.
        output_audio_dir (str): Directory to save processed audio segments.
        segment_duration (int): Duration of each segment in seconds.
    """
    # Group video files by base name (ignoring .fXXX)
    video_files = [f for f in os.listdir(input_dir) if f.endswith((".mp4", ".mkv", ".webm"))]
    audio_files = [f for f in os.listdir(input_dir) if f.endswith(".mp3")]

    # Create a mapping of base names to video files
    video_map = {}
    for video_file in video_files:
        base_name = re.sub(r"\.f\d+", "", os.path.splitext(video_file)[0])  # Strip .fXXX
        if base_name not in video_map:
            video_map[base_name] = []
        video_map[base_name].append(video_file)

    for base_name, files in video_map.items():
        # Select the best quality video
        selected_video = select_best_quality(files)
        video_path = os.path.join(input_dir, selected_video)

        # Find corresponding audio
        audio_file = f"{base_name}.mp3"
        audio_path = os.path.join(input_dir, audio_file)

        if not os.path.exists(audio_path):
            print(f"Skipping {selected_video}: No corresponding audio file found.")
            continue

        # Standardize video format to .mp4
        standardized_video_path = os.path.join(input_dir, f"{os.path.splitext(selected_video)[0]}.mp4")
        if not selected_video.endswith(".mp4"):
            convert_video_to_mp4(video_path, standardized_video_path)
        else:
            standardized_video_path = video_path

        # Process video and audio into 10-second segments
        process_video(standardized_video_path, output_video_dir, segment_duration)
        process_audio(audio_path, output_audio_dir, segment_duration)

def select_best_quality(files):
    """
    Select the best-quality video file from a list of candidates.

    Args:
        files (list): List of video filenames.

    Returns:
        str: The filename of the best-quality video.
    """
    def get_quality_score(filename):
        match = re.search(r"f(\d+)", filename)  # Look for fXXX pattern
        return int(match.group(1)) if match else 0  # Default to 0 if no match

    # Prioritize MP4 files, then sort by quality score
    mp4_files = [f for f in files if f.endswith(".mp4")]
    other_files = [f for f in files if not f.endswith(".mp4")]

    if mp4_files:
        return sorted(mp4_files, key=get_quality_score, reverse=True)[0]
    return sorted(other_files, key=get_quality_score, reverse=True)[0]

def convert_video_to_mp4(input_path, output_path):
    """
    Convert a video to .mp4 format using ffmpeg.

    Args:
        input_path (str): Path to the input video.
        output_path (str): Path to save the converted video.
    """
    subprocess.run([
        "ffmpeg", "-i", input_path, "-vcodec", "libx264", "-crf", "23", "-preset", "medium", output_path
    ], check=True)

def process_video(video_path, output_dir, segment_duration):
    """
    Splits a video into 10-second segments, excluding the first 10 seconds.

    Args:
        video_path (str): Path to the input video.
        output_dir (str): Directory to save video segments.
        segment_duration (int): Duration of each segment in seconds.
    """
    cap = cv2.VideoCapture(video_path)
    fps = 16
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    num_segments = int((duration - 10) // segment_duration)  # Number of 10-second segments
    start_frame = int(fps * 10)  # Skip the first 10 seconds

    for i in range(num_segments):
        segment_start = start_frame + i * segment_duration * fps
        segment_end = segment_start + segment_duration * fps
        segment_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_segment_{i + 1}.mp4")
        save_video_segment(cap, segment_start, segment_end, fps, segment_path)

    cap.release()

def save_video_segment(cap, start_frame, end_frame, fps, output_path):
    """
    Saves a segment of a video to a file, resizing frames to 224x224.

    Args:
        cap (cv2.VideoCapture): OpenCV video capture object.
        start_frame (int): Starting frame of the segment.
        end_frame (int): Ending frame of the segment.
        fps (int): Frames per second of the video.
        output_path (str): Path to save the video segment.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width, height = 224, 224  # Target resolution
    out = cv2.VideoWriter(output_path, fourcc, 16, (width, height))

    for _ in range(int(end_frame - start_frame)):
        ret, frame = cap.read()
        if not ret:
            break
        # Resize frame to 224x224
        resized_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        out.write(resized_frame)

    out.release()

def process_audio(audio_path, output_dir, segment_duration):
    """
    Splits an audio file into 10-second segments, excluding the first 10 seconds.

    Args:
        audio_path (str): Path to the input audio.
        output_dir (str): Directory to save audio segments.
        segment_duration (int): Duration of each segment in seconds.
    """
    audio, sr = librosa.load(audio_path, sr=32000)
    total_samples = len(audio)
    duration = total_samples / sr

    num_segments = int((duration - 10) // segment_duration)  # Number of 10-second segments
    start_sample = int(sr * 10)  # Skip the first 10 seconds

    for i in range(num_segments):
        segment_start = start_sample + i * segment_duration * sr
        segment_end = segment_start + segment_duration * sr
        segment_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_path))[0]}_segment_{i + 1}.mp3")
        sf.write(segment_path, audio[int(segment_start):int(segment_end)], sr)

if __name__ == "__main__":
    data_dir = "data"
    output_video_dir = "dat/processed_videos"
    output_audio_dir = "dat/processed_audios"

    preprocess_instrument_folders(data_dir, output_video_dir, output_audio_dir)
