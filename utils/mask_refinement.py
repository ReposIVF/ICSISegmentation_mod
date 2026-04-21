"""
Post-processing refinement for YOLO segmentation masks.

Improvements:
  - Fill small holes in masks (noise reduction)
  - Close gaps in mask boundaries (connectivity)
  - Remove small isolated regions (morphological opening)
  - Smooth mask edges (optional)
  - Extract largest contour if multiple exist
"""
import numpy as np
import cv2
from typing import Tuple, Optional


class MaskRefinementConfig:
    """Configuration for mask refinement operations."""
    
    def __init__(
        self,
        enable_close: bool = True,
        close_kernel_size: int = 3,
        close_iterations: int = 1,
        
        enable_open: bool = True,
        open_kernel_size: int = 3,
        open_iterations: int = 1,
        
        enable_fill_holes: bool = True,
        min_hole_area: int = 10,
        
        enable_smooth: bool = False,
        smooth_kernel_size: int = 5,
    ):
        """
        Args:
            enable_close        : Morphological closing (dilate then erode)
            close_kernel_size   : Size of closing kernel
            close_iterations    : Number of closing iterations
            
            enable_open         : Morphological opening (erode then dilate)
            open_kernel_size    : Size of opening kernel
            open_iterations     : Number of opening iterations
            
            enable_fill_holes    : Fill small holes in mask
            min_hole_area       : Minimum hole area to fill (pixels²)
            
            enable_smooth        : Smooth mask edges (Gaussian blur + threshold)
            smooth_kernel_size  : Size of smoothing kernel
        """
        self.enable_close = enable_close
        self.close_kernel_size = close_kernel_size
        self.close_iterations = close_iterations
        
        self.enable_open = enable_open
        self.open_kernel_size = open_kernel_size
        self.open_iterations = open_iterations
        
        self.enable_fill_holes = enable_fill_holes
        self.min_hole_area = min_hole_area
        
        self.enable_smooth = enable_smooth
        self.smooth_kernel_size = smooth_kernel_size


def close_mask(
    mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
) -> np.ndarray:
    """
    Morphological closing: dilate then erode.
    
    Closes small gaps and fills small holes in the mask.
    
    Args:
        mask       : Binary mask (0 or >0)
        kernel_size: Size of morphological kernel (must be odd)
        iterations : Number of iterations
    
    Returns:
        Refined binary mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def open_mask(
    mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
) -> np.ndarray:
    """
    Morphological opening: erode then dilate.
    
    Removes small isolated noise regions while preserving larger features.
    
    Args:
        mask       : Binary mask (0 or >0)
        kernel_size: Size of morphological kernel (must be odd)
        iterations : Number of iterations
    
    Returns:
        Refined binary mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)


def fill_holes(
    mask: np.ndarray,
    min_hole_area: int = 10,
) -> np.ndarray:
    """
    Fill small holes inside the mask.
    
    Uses contour-based hole filling to close internal voids.
    
    Args:
        mask          : Binary mask (0 or >0)
        min_hole_area : Minimum hole area (pixels²) to fill
    
    Returns:
        Refined binary mask with holes filled
    """
    # Make a copy to avoid modifying input
    refined = mask.copy()
    
    # Find all contours (including holes)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if hierarchy is None or len(contours) <= 1:
        return refined
    
    # Iterate through contours and fill holes (child contours of the main sperm)
    hierarchy = hierarchy[0]
    
    for idx, h in enumerate(hierarchy):
        parent = h[3]  # Index of parent contour
        
        # If this is a hole (has a parent), and it's small enough, fill it
        if parent >= 0:
            area = cv2.contourArea(contours[idx])
            if area < min_hole_area:
                cv2.drawContours(refined, [contours[idx]], 0, 255, -1)
    
    return refined


def smooth_mask_edges(
    mask: np.ndarray,
    kernel_size: int = 5,
) -> np.ndarray:
    """
    Smooth mask edges using Gaussian blur followed by binary threshold.
    
    Creates smoother, less jagged boundaries.
    
    Args:
        mask       : Binary mask (0 or >0)
        kernel_size: Size of Gaussian kernel (must be odd)
    
    Returns:
        Refined binary mask with smoother edges
    """
    # Gaussian blur to smooth
    blurred = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    
    # Re-binarize at 127 threshold
    _, refined = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    
    return refined


def extract_largest_contour(mask: np.ndarray) -> np.ndarray:
    """
    If mask has multiple disconnected regions, keep only the largest.
    
    Useful for removing small noise regions that YOLO might have detected.
    
    Args:
        mask : Binary mask (0 or >0)
    
    Returns:
        Refined binary mask with only largest contour
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return mask
    
    if len(contours) == 1:
        return mask  # Already single contour
    
    # Find largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create new mask with only the largest contour
    refined = np.zeros_like(mask)
    cv2.drawContours(refined, [largest_contour], 0, 255, -1)
    
    return refined


def refine_mask(
    mask: np.ndarray,
    config: Optional[MaskRefinementConfig] = None,
) -> np.ndarray:
    """
    Apply all configured refinement operations to a mask.
    
    Args:
        mask   : Binary mask from YOLO (0 or >0)
        config : MaskRefinementConfig object (uses defaults if None)
    
    Returns:
        Refined binary mask
    """
    if config is None:
        config = MaskRefinementConfig()
    
    refined = mask.copy()
    
    # Ensure binary
    refined = (refined > 0).astype(np.uint8) * 255
    
    # 1. Extract largest contour only (removes noise blobs)
    refined = extract_largest_contour(refined)
    
    # 2. Morphological closing (fill small gaps)
    if config.enable_close:
        refined = close_mask(
            refined,
            kernel_size=config.close_kernel_size,
            iterations=config.close_iterations
        )
    
    # 3. Fill holes
    if config.enable_fill_holes:
        refined = fill_holes(refined, min_hole_area=config.min_hole_area)
    
    # 4. Morphological opening (remove small noise)
    if config.enable_open:
        refined = open_mask(
            refined,
            kernel_size=config.open_kernel_size,
            iterations=config.open_iterations
        )
    
    # 5. Edge smoothing (optional)
    if config.enable_smooth:
        refined = smooth_mask_edges(refined, kernel_size=config.smooth_kernel_size)
    
    return refined


def batch_refine_masks(
    masks: dict[int, np.ndarray],
    config: Optional[MaskRefinementConfig] = None,
) -> dict[int, np.ndarray]:
    """
    Refine multiple masks efficiently.
    
    Args:
        masks  : Dictionary mapping mask_id -> binary mask array
        config : MaskRefinementConfig (uses defaults if None)
    
    Returns:
        Dictionary mapping mask_id -> refined binary mask
    """
    refined_masks = {}
    for mask_id, mask in masks.items():
        refined_masks[mask_id] = refine_mask(mask, config)
    
    return refined_masks


def compute_mask_quality_metrics(
    original_mask: np.ndarray,
    refined_mask: np.ndarray,
) -> dict:
    """
    Compute metrics comparing original and refined masks.
    
    Useful for understanding what refinement achieved.
    
    Returns:
        Dictionary with metrics
    """
    orig_area = np.sum(original_mask > 0)
    refined_area = np.sum(refined_mask > 0)
    
    # Dice coefficient (overlap)
    intersection = np.sum((original_mask > 0) & (refined_mask > 0))
    dice = 2 * intersection / (orig_area + refined_area) if (orig_area + refined_area) > 0 else 0
    
    # Area change
    area_change_pct = (refined_area - orig_area) / orig_area * 100 if orig_area > 0 else 0
    
    # Contour smoothness (via perimeter to area ratio)
    orig_contours, _ = cv2.findContours(
        (original_mask > 0).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    refined_contours, _ = cv2.findContours(
        refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    orig_perimeter = cv2.arcLength(orig_contours[0], True) if orig_contours else 0
    refined_perimeter = cv2.arcLength(refined_contours[0], True) if refined_contours else 0
    
    return {
        "original_area_pixels": int(orig_area),
        "refined_area_pixels": int(refined_area),
        "area_change_percent": area_change_pct,
        "dice_coefficient": dice,
        "original_perimeter": orig_perimeter,
        "refined_perimeter": refined_perimeter,
        "perimeter_change_percent": (
            (refined_perimeter - orig_perimeter) / orig_perimeter * 100
            if orig_perimeter > 0 else 0
        ),
    }


def apply_refinement_with_logging(
    mask: np.ndarray,
    sperm_id: int = None,
    config: Optional[MaskRefinementConfig] = None,
    logger = None,
) -> np.ndarray:
    """
    Refine a mask and log the improvement metrics.
    
    Args:
        mask     : Original binary mask
        sperm_id : For logging context
        config   : MaskRefinementConfig
        logger   : Logger instance (if None, no logging)
    
    Returns:
        Refined binary mask
    """
    refined = refine_mask(mask, config)
    
    if logger is not None:
        metrics = compute_mask_quality_metrics(mask, refined)
        id_str = f"sperm_{sperm_id}" if sperm_id is not None else "mask"
        
        logger.debug(
            f"Mask refinement for {id_str}: area {metrics['original_area_pixels']} → "
            f"{metrics['refined_area_pixels']} px², "
            f"Dice: {metrics['dice_coefficient']:.3f}"
        )
    
    return refined
