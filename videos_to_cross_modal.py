import os
import shutil
import subprocess

def videos_to_cross_modal(input_dir, output_video_dir, output_audio_dir, audio_format="mp3"):
    """
    Extracts audio from videos and separates them into two directories.

    Args:
        input_dir (str): Directory containing the input videos.
        output_video_dir (str): Directory to store the original videos.
        output_audio_dir (str): Directory to store the extracted audio files.
        audio_format (str): Format for the extracted audio (default: "mp3").

    Returns:
        None
    """
    # Create output directories if they don't exist
    os.makedirs(output_video_dir, exist_ok=True)
    os.makedirs(output_audio_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith((".mp4", ".avi", ".mov")):  # Supported video formats
            video_path = os.path.join(input_dir, file)
            video_output_path = os.path.join(output_video_dir, os.path.basename(file))
            audio_output_path = os.path.join(output_audio_dir, f"{os.path.splitext(file)[0]}.{audio_format}")

            # Copy the video to the output video directory
            shutil.copy(video_path, video_output_path)

            # Extract audio and save in the specified format
            extract_audio(video_path, audio_output_path)
            print(f"Processed: {file}")

def extract_audio(video_path, audio_output_path):
    """
    Extracts audio from a video using ffmpeg.

    Args:
        video_path (str): Path to the input video.
        audio_output_path (str): Path to save the extracted audio.

    Returns:
        None
    """
    subprocess.run([
        "ffmpeg", "-i", video_path, "-vn",  # -vn ensures no video in output
        "-ac", "2", "-ar", "32000", "-b:a", "192k",  # Stereo, 44.1 kHz, 192 kbps
        audio_output_path
    ], check=True)

if __name__ == "__main__":
    # Paths for testing
    input_dir = "testda"  # Folder with test videos
    output_video_dir = "test_dat/videos"
    output_audio_dir = "test_dat/audios"
#
    videos_to_cross_modal(input_dir, output_video_dir, output_audio_dir)
