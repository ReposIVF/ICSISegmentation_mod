"""
Frame-level quality validation for the tracking pipeline.

Detects:
  - Motion blur (via Laplacian variance)
  - Oversaturation (clipped pixel values)
  - Undersaturation (low signal)
  - Unusual mask coverage
  - Potential frame corruption
"""
import numpy as np
import cv2
from typing import Tuple, Dict, List


# ── Default Quality Thresholds ──
BLUR_VARIANCE_THRESHOLD = 100.0  # Lower = blurrier
SATURATION_RATIO_HIGH = 0.10  # >10% saturated = warning
SATURATION_RATIO_LOW = 0.5   # >50% of pixels min value = warning
MASK_COVERAGE_MIN = 0.0005   # At least 0.05% should be mask
MASK_COVERAGE_MAX = 0.5      # At most 50% should be mask (something wrong)


class FrameQualityReport:
    """Result of frame quality validation."""
    
    def __init__(self):
        self.is_valid = True
        self.issues: List[Dict] = []  # List of {"type", "severity", "details"}
    
    def add_issue(self, issue_type: str, severity: str, details: str):
        """Add a quality issue."""
        self.issues.append({
            "type": issue_type,
            "severity": severity,  # "low", "medium", "high"
            "details": details,
        })
        if severity == "high":
            self.is_valid = False
    
    def __str__(self) -> str:
        """Pretty print quality report."""
        if not self.issues:
            return "✅ Frame quality: GOOD"
        
        lines = [f"⚠️  Frame quality issues detected:"]
        for issue in self.issues:
            severity_icon = "🔴" if issue["severity"] == "high" else "🟡"
            lines.append(f"  {severity_icon} {issue['type']}: {issue['details']}")
        return "\n".join(lines)


def check_blur(frame: np.ndarray, threshold: float = BLUR_VARIANCE_THRESHOLD) -> Tuple[float, bool]:
    """
    Detect motion blur using Laplacian variance.
    
    Returns:
        (laplacian_variance, is_acceptable)
        - Higher variance = sharper image
        - Lower variance = blurrier image
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    
    is_acceptable = variance >= threshold
    
    return variance, is_acceptable


def check_saturation(frame: np.ndarray) -> Tuple[float, float, bool]:
    """
    Detect oversaturation (clipped highlights) and undersaturation (low signal).
    
    Returns:
        (saturation_high_ratio, saturation_low_ratio, is_acceptable)
    """
    if len(frame.shape) == 3:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
    else:
        v_channel = frame
    
    total_pixels = v_channel.size
    
    # High saturation: V channel at max (255) = clipped signals
    saturated_high = np.sum(v_channel == 255) / total_pixels
    
    # Low saturation: V channel at or near min (0-20) = no signal
    saturated_low = np.sum(v_channel <= 20) / total_pixels
    
    is_acceptable = (
        saturated_high < SATURATION_RATIO_HIGH and
        saturated_low < SATURATION_RATIO_LOW
    )
    
    return saturated_high, saturated_low, is_acceptable


def check_mask_coverage(mask: np.ndarray) -> Tuple[float, bool]:
    """
    Verify mask coverage is reasonable (not all black, not all white).
    
    Returns:
        (coverage_ratio, is_acceptable)
    """
    if mask.size == 0:
        return 0.0, False
    
    coverage = np.sum(mask > 0) / mask.size
    is_acceptable = MASK_COVERAGE_MIN <= coverage <= MASK_COVERAGE_MAX
    
    return coverage, is_acceptable


def check_frame_content_stability(frame: np.ndarray, prev_frame: np.ndarray = None) -> Tuple[float, bool]:
    """
    Detect abrupt frame changes or potential corruption via histogram comparison.
    
    Returns:
        (histogram_correlation, is_stable)
        - Correlation close to 1.0 = similar frames
        - Correlation close to 0.0 = very different frames (possible corruption)
    """
    if prev_frame is None:
        return 1.0, True
    
    if frame.shape != prev_frame.shape:
        return 0.0, False
    
    if len(frame.shape) == 3:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    else:
        frame_gray = frame
        prev_gray = prev_frame
    
    hist_curr = cv2.calcHist([frame_gray], [0], None, [256], [0, 256])
    hist_prev = cv2.calcHist([prev_gray], [0], None, [256], [0, 256])
    
    correlation = cv2.compareHist(hist_curr, hist_prev, cv2.HISTCMP_CORREL)
    
    # Assume stability if correlation > 0.3 (allow some motion)
    is_stable = correlation > 0.3
    
    return correlation, is_stable


def validate_frame_quality(
    frame: np.ndarray,
    mask: np.ndarray = None,
    prev_frame: np.ndarray = None,
    frame_number: int = 0,
    verbose: bool = False,
) -> FrameQualityReport:
    """
    Comprehensive frame quality validation.
    
    Args:
        frame        : Current frame (BGR or grayscale)
        mask         : Optional detection mask from YOLO
        prev_frame   : Optional previous frame for stability check
        frame_number : For debugging info
        verbose      : If True, return detailed report
    
    Returns:
        FrameQualityReport object
    """
    report = FrameQualityReport()
    
    # ── Blur Check ──
    laplacian_var, blur_ok = check_blur(frame)
    if not blur_ok:
        severity = "high" if laplacian_var < 50 else "medium"
        report.add_issue(
            "blur_detected",
            severity,
            f"Laplacian variance={laplacian_var:.1f} (threshold={BLUR_VARIANCE_THRESHOLD})"
        )
    
    # ── Saturation Check ──
    sat_high, sat_low, sat_ok = check_saturation(frame)
    if not sat_ok:
        if sat_high > SATURATION_RATIO_HIGH:
            report.add_issue(
                "over_saturation",
                "high",
                f"{sat_high*100:.1f}% pixels clipped (threshold={SATURATION_RATIO_HIGH*100}%)"
            )
        if sat_low > SATURATION_RATIO_LOW:
            report.add_issue(
                "under_saturation",
                "medium",
                f"{sat_low*100:.1f}% pixels near-black (threshold={SATURATION_RATIO_LOW*100}%)"
            )
    
    # ── Mask Coverage Check ──
    if mask is not None:
        coverage, coverage_ok = check_mask_coverage(mask)
        if not coverage_ok:
            if coverage < MASK_COVERAGE_MIN:
                report.add_issue(
                    "low_mask_coverage",
                    "medium",
                    f"{coverage*100:.3f}% masked (min threshold={MASK_COVERAGE_MIN*100:.3f}%)"
                )
            elif coverage > MASK_COVERAGE_MAX:
                report.add_issue(
                    "high_mask_coverage",
                    "high",
                    f"{coverage*100:.1f}% masked (max threshold={MASK_COVERAGE_MAX*100}%)"
                )
    
    # ── Stability Check ──
    if prev_frame is not None:
        correlation, stable_ok = check_frame_content_stability(frame, prev_frame)
        if not stable_ok:
            report.add_issue(
                "unstable_frames",
                "high",
                f"Histogram correlation={correlation:.2f} (indicates possible corruption or jump cut)"
            )
    
    return report


def get_frame_quality_metrics(frame: np.ndarray) -> Dict[str, float]:
    """
    Extract all quality metrics for a frame (without validation).
    Useful for logging and analysis.
    
    Returns:
        Dictionary of metrics
    """
    laplacian_var, _ = check_blur(frame)
    sat_high, sat_low, _ = check_saturation(frame)
    
    if len(frame.shape) == 3:
        h, w, c = frame.shape
    else:
        h, w = frame.shape
    
    return {
        "laplacian_variance": laplacian_var,
        "saturation_high_ratio": sat_high,
        "saturation_low_ratio": sat_low,
        "height_pixels": h,
        "width_pixels": w,
    }
