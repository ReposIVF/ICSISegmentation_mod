"""
Centralized logging system for the sperm tracking pipeline.

Features:
  - Console output (INFO level and above)
  - File logging with rotation (DEBUG level and above)
  - Structured metrics logging to CSV
  - Pipeline-wide logger available via get_logger()
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


# Global logger instance
_logger = None


def setup_logger(log_dir: str = "./logs", name: str = "sperm_pipeline", level=logging.DEBUG):
    """
    Initialize the global logger with console and file handlers.
    
    Args:
        log_dir  : Directory to store log files
        name     : Logger name
        level    : Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logging.Logger instance
    """
    global _logger
    
    if _logger is not None:
        return _logger
    
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    _logger = logging.getLogger(name)
    _logger.setLevel(level)
    
    # Prevent adding duplicate handlers
    if _logger.hasHandlers():
        return _logger
    
    # ── Console Handler (INFO and above) ──
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - [%(levelname)-8s] - %(name)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # ── File Handler (DEBUG and above, with rotation) ──
    log_file = log_path / f"{name}.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - [%(levelname)-8s] - %(name)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # ── Add handlers to logger ──
    _logger.addHandler(console_handler)
    _logger.addHandler(file_handler)
    
    _logger.info(f"Logger initialized. Log file: {log_file}")
    
    return _logger


def get_logger() -> logging.Logger:
    """
    Get the global logger instance.
    
    If not initialized, initializes with default settings.
    Always safe to call.
    """
    global _logger
    
    if _logger is None:
        setup_logger()
    
    return _logger


class PipelineMetrics:
    """
    Collects and manages pipeline performance metrics.
    """
    
    def __init__(self):
        self.frame_count = 0
        self.total_detections = 0
        self.detections_per_frame = []
        self.frame_quality_issues = []  # List of (frame_no, issue_type, severity)
        self.feature_validation_failures = []  # List of (sperm_id, feature, reason)
        self.filtered_counts = {
            "hard_filters": 0,
            "static_filter": 0,
            "soft_filters": 0,
            "morpho_penalties": 0,
        }
        self.active_track_ids = set()
        self.track_id_switches = 0
        self.inference_times = []  # Frames/sec measurements
        
    def log_detection(self, sperm_id: int, score: float):
        """Record a detection."""
        self.total_detections += 1
        self.active_track_ids.add(sperm_id)
    
    def log_filter(self, filter_type: str, sperm_id: int, reason: str = ""):
        """Record that a sperm was filtered out."""
        if filter_type in self.filtered_counts:
            self.filtered_counts[filter_type] += 1
            logger = get_logger()
            logger.debug(f"Sperm {sperm_id} filtered by {filter_type}: {reason}")
    
    def log_feature_validation_failure(self, sperm_id: int, feature: str, reason: str):
        """Record feature validation failure."""
        self.feature_validation_failures.append((sperm_id, feature, reason))
        logger = get_logger()
        logger.warning(f"Feature validation failed for sperm {sperm_id}: {feature} - {reason}")
    
    def log_frame_quality_issue(self, frame_no: int, issue_type: str, severity: str, details: str = ""):
        """Record frame quality issue."""
        self.frame_quality_issues.append((frame_no, issue_type, severity, details))
        logger = get_logger()
        log_level = logging.WARNING if severity == "high" else logging.DEBUG
        logger.log(log_level, f"Frame {frame_no} quality issue ({issue_type}): {details}")
    
    def summary(self) -> dict:
        """Return summary of metrics."""
        avg_detections = (
            sum(self.detections_per_frame) / len(self.detections_per_frame)
            if self.detections_per_frame else 0
        )
        
        return {
            "total_frames": self.frame_count,
            "total_detections": self.total_detections,
            "avg_detections_per_frame": avg_detections,
            "active_track_ids": len(self.active_track_ids),
            "track_id_switches": self.track_id_switches,
            "frame_quality_issues": len(self.frame_quality_issues),
            "feature_validation_failures": len(self.feature_validation_failures),
            "filtered_hard": self.filtered_counts["hard_filters"],
            "filtered_static": self.filtered_counts["static_filter"],
            "filtered_soft": self.filtered_counts["soft_filters"],
            "filtered_morpho": self.filtered_counts["morpho_penalties"],
        }
    
    def print_summary(self):
        """Print formatted summary to console."""
        logger = get_logger()
        summary = self.summary()
        
        logger.info("=" * 70)
        logger.info("PIPELINE METRICS SUMMARY")
        logger.info("=" * 70)
        logger.info(f"  Total frames processed:        {summary['total_frames']}")
        logger.info(f"  Total detections:              {summary['total_detections']}")
        logger.info(f"  Avg detections/frame:          {summary['avg_detections_per_frame']:.1f}")
        logger.info(f"  Active track IDs:              {summary['active_track_ids']}")
        logger.info(f"  Frame quality issues:          {summary['frame_quality_issues']}")
        logger.info(f"  Feature validation failures:   {summary['feature_validation_failures']}")
        logger.info(f"  Sperm filtered (hard):         {summary['filtered_hard']}")
        logger.info(f"  Sperm filtered (static):       {summary['filtered_static']}")
        logger.info(f"  Sperm filtered (soft):         {summary['filtered_soft']}")
        logger.info(f"  Sperm filtered (morpho):       {summary['filtered_morpho']}")
        logger.info("=" * 70)
