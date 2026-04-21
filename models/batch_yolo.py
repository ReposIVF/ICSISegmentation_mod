"""
Batch YOLO inference wrapper for improved throughput.

Supports:
  - Frame batching (multiple frames processed together)
  - Configurable batch sizes based on available GPU/CPU memory
  - Frame buffering with callback processing
  - Per-frame result unpacking
  - Inference timing metrics

Note: Ultralytics 8.0.220+ supports batch inference via predict(source=[...])
"""
import numpy as np
from typing import List, Callable, Optional, Dict, Any
from collections import deque
import time


class BatchYOLOProcessor:
    """
    Batches video frames for efficient YOLO inference.
    
    Usage:
        processor = BatchYOLOProcessor(
            yolo_model=model,
            batch_size=16,
            callback=process_results
        )
        
        for frame in video:
            processor.add_frame(frame, frame_idx)
        
        processor.flush()  # Process remaining frames
    """
    
    def __init__(
        self,
        yolo_model,
        batch_size: int = 16,
        callback: Optional[Callable] = None,
        conf: float = 0.7,
        classes: Optional[List[int]] = None,
    ):
        """
        Args:
            yolo_model   : Ultralytics YOLO model (from load_yolo)
            batch_size   : Number of frames to accumulate before inference
            callback     : Function called with (results, frame_indices) after inference
            conf         : Detection confidence threshold
            classes      : List of class IDs to detect (None = all)
        """
        self.model = yolo_model
        self.batch_size = batch_size
        self.callback = callback
        self.conf = conf
        self.classes = classes
        
        self.frame_buffer = deque()      # Stores (frame, frame_idx) tuples
        self.results_map = {}            # Maps frame_idx -> results
        
        # Timing metrics
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.total_frames_processed = 0
    
    def add_frame(self, frame: np.ndarray, frame_idx: int):
        """
        Add a frame to the batch queue.
        
        If batch is full, automatically triggers inference.
        
        Args:
            frame     : Video frame (BGR)
            frame_idx : Frame number for tracking
        """
        self.frame_buffer.append((frame, frame_idx))
        
        if len(self.frame_buffer) >= self.batch_size:
            self.process_batch()
    
    def process_batch(self) -> Dict[int, Any]:
        """
        Process accumulated frames as a batch.
        
        Returns:
            Dictionary mapping frame_idx -> results
        """
        if len(self.frame_buffer) == 0:
            return {}
        
        # ── Prepare batch ──
        frames = []
        frame_indices = []
        
        while self.frame_buffer:
            frame, frame_idx = self.frame_buffer.popleft()
            frames.append(frame)
            frame_indices.append(frame_idx)
        
        # ── Run YOLO on batch ──
        t_start = time.time()
        
        results_batch = self.model.predict(
            source=frames,
            conf=self.conf,
            classes=self.classes if self.classes else 0,  # Sperm only
            verbose=False,
        )
        
        t_end = time.time()
        inference_time = t_end - t_start
        
        # ── Store results and map indices ──
        batch_results = {}
        for frame_idx, results in zip(frame_indices, results_batch):
            self.results_map[frame_idx] = results
            batch_results[frame_idx] = results
        
        # ── Update metrics ──
        self.total_frames_processed += len(frames)
        self.total_inference_time += inference_time
        self.inference_count += 1
        
        # ── Call callback if provided ──
        if self.callback is not None:
            self.callback(batch_results, frame_indices)
        
        return batch_results
    
    def flush(self) -> Dict[int, Any]:
        """
        Process remaining frames in buffer.
        
        Call at end of video to ensure all frames are processed.
        
        Returns:
            Dictionary mapping frame_idx -> results
        """
        return self.process_batch()
    
    def get_results(self, frame_idx: int):
        """Retrieve results for a specific frame."""
        return self.results_map.get(frame_idx, None)
    
    def get_inference_fps(self) -> float:
        """
        Calculate average frames per second during inference.
        
        Returns:
            Frames/sec (higher is better)
        """
        if self.total_inference_time == 0:
            return 0.0
        
        return self.total_frames_processed / self.total_inference_time
    
    def get_metrics(self) -> Dict[str, float]:
        """Get inference timing metrics."""
        return {
            "total_frames": self.total_frames_processed,
            "total_batches": self.inference_count,
            "total_time_sec": self.total_inference_time,
            "avg_batch_size": (
                self.total_frames_processed / self.inference_count
                if self.inference_count > 0 else 0
            ),
            "inference_fps": self.get_inference_fps(),
            "avg_time_per_batch_sec": (
                self.total_inference_time / self.inference_count
                if self.inference_count > 0 else 0
            ),
        }
    
    def reset_metrics(self):
        """Reset all timing metrics."""
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.total_frames_processed = 0


def create_batch_processor(
    yolo_model,
    batch_size: int = 16,
    conf: float = 0.7,
    callback: Optional[Callable] = None,
) -> BatchYOLOProcessor:
    """
    Factory function to create a batch processor.
    
    Args:
        yolo_model : Loaded YOLO model
        batch_size : Frames per batch
        conf       : Detection confidence threshold
        callback   : Optional function to call after each batch
    
    Returns:
        BatchYOLOProcessor instance
    """
    return BatchYOLOProcessor(
        yolo_model=yolo_model,
        batch_size=batch_size,
        callback=callback,
        conf=conf,
        classes=[0],  # Sperm detection only
    )


def adaptive_batch_size(
    available_memory_mb: float = 2048,
    frame_height: int = 1080,
    frame_width: int = 1920,
) -> int:
    """
    Estimate optimal batch size based on available GPU/CPU memory.
    
    This is a heuristic; may need tuning for specific hardware.
    
    Args:
        available_memory_mb : Available GPU/CPU memory in MB
        frame_height        : Frame height in pixels
        frame_width         : Frame width in pixels
    
    Returns:
        Recommended batch size
    """
    # Rough estimate: YOLO detection ~25-50 MB per frame depending on model
    # (with overhead, input buffer, output buffer)
    bytes_per_frame = (frame_height * frame_width * 3 * 1.5) / (1024 * 1024)  # ~9 MB for 1080p
    model_overhead_mb = 50  # Model weights + buffers
    
    available_for_frames = available_memory_mb - model_overhead_mb
    
    batch_size = max(1, int(available_for_frames / bytes_per_frame))
    
    # Conservative cap
    batch_size = min(batch_size, 64)
    
    return batch_size


class SequentialYOLOProcessor:
    """
    Fallback wrapper that processes frames one-by-one (original behavior).
    
    Used for compatibility or when batching is not beneficial.
    """
    
    def __init__(
        self,
        yolo_model,
        conf: float = 0.7,
        classes: Optional[List[int]] = None,
    ):
        self.model = yolo_model
        self.conf = conf
        self.classes = classes if classes else [0]
        
        self.results_map = {}
        self.inference_times = []
    
    def add_frame(self, frame: np.ndarray, frame_idx: int):
        """
        Process a single frame immediately.
        
        Args:
            frame     : Video frame (BGR)
            frame_idx : Frame number
        """
        t_start = time.time()
        
        results = self.model.predict(
            source=[frame],
            conf=self.conf,
            classes=self.classes[0] if self.classes else 0,
            verbose=False,
        )
        
        t_end = time.time()
        self.inference_times.append(t_end - t_start)
        
        self.results_map[frame_idx] = results[0]
    
    def get_results(self, frame_idx: int):
        """Retrieve results for a specific frame."""
        return self.results_map.get(frame_idx, None)
    
    def process_batch(self):
        """No-op for compatibility."""
        pass
    
    def flush(self):
        """No-op for compatibility."""
        pass
    
    def get_inference_fps(self) -> float:
        """Calculate average FPS."""
        if not self.inference_times:
            return 0.0
        
        avg_time = np.mean(self.inference_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def get_metrics(self) -> Dict[str, float]:
        """Get inference metrics."""
        return {
            "total_frames": len(self.results_map),
            "inference_fps": self.get_inference_fps(),
            "mode": "sequential",
        }
