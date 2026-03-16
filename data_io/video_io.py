"""
Helpers for opening video input and creating the annotated output writer.
"""
import os
import cv2


def open_video(video_path: str) -> tuple[cv2.VideoCapture, int, int, int]:
    """
    Open a video file and return the capture object plus basic metadata.

    Returns:
        cap     : cv2.VideoCapture
        width   : frame width in pixels
        height  : frame height in pixels
        fps     : frames per second (int)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"Opened: {video_path}  |  {width}x{height} @ {fps} fps")
    return cap, width, height, fps


def create_writer(video_path: str, output_dir: str, width: int, height: int, fps: int) -> cv2.VideoWriter:
    """
    Create a VideoWriter that mirrors the input video's resolution/fps,
    saving the annotated output to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(video_path))
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"Output video: {output_path}")
    return writer
