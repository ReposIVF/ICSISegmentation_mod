"""
Morphological feature validation for detected sperm.

Validates:
  - Geometric bounds (area, perimeter, extent)
  - Shape descriptors (circularity, solidity, eccentricity)
  - Aspect ratios and dimensions
  - Pixel-space consistency
"""
from typing import Dict, Tuple, List, Optional
import numpy as np


# ── Default Feature Bounds ──
# These are empirically derived from typical sperm morphology
FEATURE_BOUNDS = {
    "area": (50, 5000),                    # pixels^2
    "perimeter": (20, 300),                # pixels
    "circularity": (0.01, 1.0),           # unitless (4π*A/P²)
    "solidity": (0.3, 1.0),               # unitless (A/ConvexHull area)
    "extend": (0.1, 1.0),                 # unitless (A/BoundingBox area)
    "eccentricity": (0.0, 0.99),          # unitless (0=circle, 1=line)
    "convexity": (0.5, 1.5),              # unitless (Perimeter/HullPerimeter)
    "aspect_ratio": (0.1, 5.0),           # unitless (major/minor axis)
    "orientated_angle": (-180, 180),      # degrees
    "hull_area": (50, 6000),              # pixels^2
    "hull_perimeter": (20, 400),          # pixels
    "compactness": (0.0, 2.0),            # unitless
    "major_axis_radius": (5, 100),        # pixels
    "minor_axis_radius": (5, 100),        # pixels
}

# Velocity bounds (pixels/sec)
VELOCITY_BOUNDS = {
    "VSL": (0, 500),
    "VCL": (0, 750),
    "HMP": (0, 100),
}

# Brightness bounds
BRIGHTNESS_BOUNDS = {
    "MeanBrightness": (0, 255),
}


class FeatureValidationResult:
    """Result of feature validation."""
    
    def __init__(self):
        self.is_valid = True
        self.failures: List[Dict] = []  # List of {"feature", "value", "bounds", "reason"}
        self.warnings: List[Dict] = []
    
    def add_failure(self, feature: str, value: float, bounds: Tuple[float, float], reason: str = ""):
        """Add a validation failure."""
        self.is_valid = False
        self.failures.append({
            "feature": feature,
            "value": value,
            "bounds": bounds,
            "reason": reason,
        })
    
    def add_warning(self, feature: str, value: float, message: str):
        """Add a warning (doesn't invalidate)."""
        self.warnings.append({
            "feature": feature,
            "value": value,
            "message": message,
        })
    
    def summary(self) -> str:
        """Get summary string."""
        if self.is_valid and not self.warnings:
            return "✅ Feature validation: VALID"
        
        lines = []
        if self.failures:
            lines.append(f"❌ {len(self.failures)} validation failure(s):")
            for f in self.failures:
                lines.append(
                    f"  - {f['feature']}: {f['value']:.2f} "
                    f"(bounds: {f['bounds'][0]:.2f}-{f['bounds'][1]:.2f}) - {f['reason']}"
                )
        
        if self.warnings:
            lines.append(f"⚠️  {len(self.warnings)} warning(s):")
            for w in self.warnings:
                lines.append(f"  - {w['feature']}: {w['value']:.2f} - {w['message']}")
        
        return "\n".join(lines)


def validate_feature_value(
    feature_name: str,
    value: float,
    bounds: Optional[Tuple[float, float]] = None,
) -> Tuple[bool, str]:
    """
    Validate a single feature against bounds.
    
    Returns:
        (is_valid, reason_if_invalid)
    """
    if bounds is None:
        bounds = FEATURE_BOUNDS.get(feature_name, (0, float('inf')))
    
    min_val, max_val = bounds
    
    if np.isnan(value):
        return False, "value is NaN"
    
    if np.isinf(value):
        return False, "value is infinite"
    
    if value < min_val:
        return False, f"below minimum ({min_val:.2f})"
    
    if value > max_val:
        return False, f"above maximum ({max_val:.2f})"
    
    return True, ""


def validate_morpho_features(features: Dict[str, float]) -> FeatureValidationResult:
    """
    Validate all morphological features for a detected sperm.
    
    Args:
        features : Dictionary of feature_name -> value
    
    Returns:
        FeatureValidationResult object
    """
    result = FeatureValidationResult()
    
    # ── Check critical features (must be valid) ──
    critical_features = [
        "area",
        "circularity",
        "solidity",
        "perimeter",
    ]
    
    for feature in critical_features:
        if feature not in features:
            result.add_failure(feature, 0, FEATURE_BOUNDS[feature], "feature not extracted")
            continue
        
        value = features[feature]
        bounds = FEATURE_BOUNDS.get(feature, (0, float('inf')))
        is_valid, reason = validate_feature_value(feature, value, bounds)
        
        if not is_valid:
            result.add_failure(feature, value, bounds, reason)
    
    # ── Check optional features (warnings only) ──
    optional_features = {
        "eccentricity": FEATURE_BOUNDS.get("eccentricity"),
        "extend": FEATURE_BOUNDS.get("extend"),
        "convexity": FEATURE_BOUNDS.get("convexity"),
        "aspect_ratio": FEATURE_BOUNDS.get("aspect_ratio"),
    }
    
    for feature, bounds in optional_features.items():
        if feature not in features:
            continue
        
        value = features[feature]
        is_valid, reason = validate_feature_value(feature, value, bounds)
        
        if not is_valid:
            result.add_warning(feature, value, f"{reason} (bounds: {bounds})")
    
    # ── Cross-feature consistency checks ──
    if "solidity" in features and "circularity" in features:
        # Very high circularity + low solidity is suspicious
        if features["circularity"] > 0.9 and features["solidity"] < 0.5:
            result.add_warning(
                "circularity_vs_solidity",
                features["circularity"],
                "High circularity but low solidity (possible noise or artifact)"
            )
    
    return result


def validate_position_data(position: Dict) -> FeatureValidationResult:
    """
    Validate a single frame position entry.
    
    Args:
        position : Dictionary with posX, posY, MeanBrightness, etc.
    
    Returns:
        FeatureValidationResult object
    """
    result = FeatureValidationResult()
    
    # ── Check position coordinates ──
    for coord_name in ["posX", "posY"]:
        if coord_name not in position:
            result.add_failure(coord_name, 0, (0, 1), "coordinate not present")
            continue
        
        value = position[coord_name]
        if not (0 <= value <= 1):
            result.add_failure(coord_name, value, (0, 1), "normalized coordinate out of range")
    
    # ── Check brightness ──
    if "MeanBrightness" in position:
        brightness = position["MeanBrightness"]
        is_valid, reason = validate_feature_value("MeanBrightness", brightness)
        
        if not is_valid:
            result.add_warning("MeanBrightness", brightness, reason)
    
    return result


def validate_velocity_metrics(velocities: Dict[str, float]) -> FeatureValidationResult:
    """
    Validate computed velocity metrics.
    
    Args:
        velocities : Dictionary with VSL, VCL, HMP
    
    Returns:
        FeatureValidationResult object
    """
    result = FeatureValidationResult()
    
    for metric_name, bounds in VELOCITY_BOUNDS.items():
        if metric_name not in velocities:
            continue
        
        value = velocities[metric_name]
        is_valid, reason = validate_feature_value(metric_name, value, bounds)
        
        if not is_valid:
            result.add_failure(metric_name, value, bounds, reason)
    
    # ── Cross-metric checks ──
    if "VCL" in velocities and "VSL" in velocities:
        # VCL should be >= VSL (curvilinear >= straight-line)
        if velocities["VCL"] < velocities["VSL"] * 0.9:  # Allow 10% tolerance
            result.add_warning(
                "VCL_vs_VSL",
                velocities["VCL"] / velocities["VSL"] if velocities["VSL"] > 0 else 0,
                "VCL < VSL (curvilinear velocity should be >= straight-line)"
            )
    
    return result


def batch_validate_features(
    features_list: List[Dict[str, float]],
    stop_on_error: bool = False,
) -> Dict[int, FeatureValidationResult]:
    """
    Validate multiple features at once.
    
    Args:
        features_list   : List of feature dictionaries
        stop_on_error   : If True, stop at first invalid feature
    
    Returns:
        Dictionary mapping index -> FeatureValidationResult
    """
    results = {}
    
    for idx, features in enumerate(features_list):
        result = validate_morpho_features(features)
        results[idx] = result
        
        if stop_on_error and not result.is_valid:
            break
    
    return results


def get_valid_features_mask(
    features_list: List[Dict[str, float]]
) -> np.ndarray:
    """
    Get boolean mask of which features are valid.
    
    Returns:
        Boolean numpy array (True = valid, False = invalid)
    """
    results = batch_validate_features(features_list)
    mask = np.array([results[i].is_valid for i in range(len(features_list))])
    return mask


def filter_valid_features(
    features_list: List[Dict[str, float]]
) -> Tuple[List[Dict[str, float]], List[int]]:
    """
    Filter out invalid features.
    
    Returns:
        (valid_features_list, original_indices_of_valid)
    """
    results = batch_validate_features(features_list)
    
    valid_features = []
    valid_indices = []
    
    for idx, result in results.items():
        if result.is_valid:
            valid_features.append(features_list[idx])
            valid_indices.append(idx)
    
    return valid_features, valid_indices
