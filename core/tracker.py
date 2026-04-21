"""
Main video tracking loop.

Responsibilities:
  - Run YOLO frame-by-frame
  - Extract morphological features from each detected mask
  - Validate frame and feature quality
  - Refine segmentation masks for better accuracy
  - Trigger live score snapshots every SCORE_WINDOW_SIZE frames
  - Delegate ranking and annotation to core.ranking
  - Write annotated frames to the VideoWriter
  - Log pipeline metrics and quality issues
"""
import math
import cv2
import numpy as np

from core.scorer import score_snapshot
from core.ranking import RankingState, draw_ranking
from utils.logger import get_logger, PipelineMetrics
from utils.frame_validation import validate_frame_quality, get_frame_quality_metrics
from utils.feature_validation import validate_morpho_features
from utils.mask_refinement import refine_mask, MaskRefinementConfig, compute_mask_quality_metrics


def extract_morpho_features(largest_contour, box, mask) -> dict:
    """
    Compute all morphological descriptors for a single contour.
    Returns a dict of raw feature values.
    """
    x_min, y_min, x_max, y_max = box
    mask_height, mask_width = mask.shape

    area = cv2.contourArea(largest_contour)
    bbox_area = (x_max - x_min) * (y_max - y_min)
    extent = float(area) / bbox_area if bbox_area > 0 else 0.0
    perimeter = cv2.arcLength(largest_contour, True)
    circularity = 4 * math.pi * area / (perimeter ** 2) if perimeter != 0 else 0.0
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    hull_perimeter = cv2.arcLength(hull, True)

    try:
        solidity = float(area) / hull_area
        convexity = perimeter / hull_perimeter
    except ZeroDivisionError:
        solidity = convexity = 0.0

    eccentricity = major_axis_radius = minor_axis_radius = 0.0
    orientation_angle = 0.0

    contour = largest_contour.get() if isinstance(largest_contour, cv2.UMat) else largest_contour
    if len(contour) >= 5:
        _, axes, orientation_angle = cv2.fitEllipse(contour)
        major_axis_length, minor_axis_length = max(axes), min(axes)
        try:
            eccentricity = np.sqrt(1 - (minor_axis_length / major_axis_length) ** 2)
            major_axis_radius = major_axis_length / 2.0
            minor_axis_radius = minor_axis_length / 2.0
        except Exception:
            pass

    aspect_ratio = float(x_max - x_min) / (y_max - y_min) if (y_max - y_min) != 0 else 0.0
    compactness = np.sqrt(4 * area / np.pi) / perimeter if perimeter != 0 else 0.0

    return {
        "area": area,
        "perimeter": perimeter,
        "aspect_ratio": aspect_ratio,
        "extend": extent,
        "orientated_angle": orientation_angle,
        "circularity": circularity,
        "hull_area": hull_area,
        "solidity": solidity,
        "hull_perimeter": hull_perimeter,
        "convexity": convexity,
        "eccentricity": eccentricity,
        "compactness": compactness,
        "major_axis_radius": major_axis_radius,
        "minor_axis_radius": minor_axis_radius,
    }


def run_tracking_loop(
    cap: cv2.VideoCapture,
    writer: cv2.VideoWriter,
    yolo_model,
    blastocyst_model,
    device: str,
    fps: int,
    width: int,
    height: int,
    morpho_params: dict | None,
    motility_params: dict | None,
    filter_cfg: dict,
    cfg: dict,
) -> dict:
    """
    Run the full frame-by-frame YOLO tracking loop with quality validation and logging.

    Returns:
        mask_info_dict : {track_id: sperm_data_dict} accumulated across all frames.
    """
    logger = get_logger()
    metrics = PipelineMetrics()
    
    tracker_config = cfg["paths"]["tracker_config"]
    padding = cfg["tracking"]["padding"]
    score_window = cfg["tracking"]["score_window_size"]
    conf = cfg["tracking"].get("conf", 0.5)
    ranking_cfg = cfg["ranking"]

    mask_info_dict: dict = {}
    ranking_state = RankingState(
        update_interval=ranking_cfg["update_interval"],
        change_threshold=ranking_cfg["change_threshold"],
        smoothing_alpha=ranking_cfg["smoothing_alpha"],
    )

    # ── Mask refinement configuration ──
    mask_refine_cfg = MaskRefinementConfig(
        enable_close=True,
        close_kernel_size=3,
        close_iterations=1,
        enable_open=True,
        open_kernel_size=3,
        open_iterations=1,
        enable_fill_holes=True,
        min_hole_area=10,
    )

    frame_number = 0
    prev_frame = None

    logger.info(f"Starting tracking loop: {width}x{height} @ {fps} fps")
    logger.info(f"Detection confidence threshold: {conf}")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_number += 1
        metrics.frame_count += 1

        # ── Frame quality validation ──
        frame_quality = validate_frame_quality(
            frame, prev_frame=prev_frame, frame_number=frame_number, verbose=False
        )
        
        if not frame_quality.is_valid:
            for issue in frame_quality.issues:
                metrics.log_frame_quality_issue(
                    frame_number, issue["type"], issue["severity"], issue["details"]
                )
        
        prev_frame = frame.copy()

        # ── YOLO Tracking ──
        try:
            results = yolo_model.track(
                frame,
                tracker=tracker_config,
                persist=True,
                show_boxes=False,
                show_labels=True,
                show=False,
                classes=0,
                conf=conf,
            )
        except Exception as e:
            logger.error(f"YOLO tracking failed on frame {frame_number}: {e}")
            continue

        if results[0].boxes and results[0].boxes.id is not None:
            boxes       = results[0].boxes.xyxyn.cpu().numpy()
            cls         = results[0].boxes.cls.cpu().numpy().astype(int)
            track_ids   = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy().astype(float)
            masks       = (results[0].masks.data.cpu().numpy() * 255).astype("uint8")
        else:
            boxes = track_ids = confidences = masks = cls = []

        annotated_frame = results[0].plot(boxes=False)
        
        metrics.detections_per_frame.append(len(track_ids))
        metrics.total_detections += len(track_ids)

        for box, track_id, mask, class_id, confidence in zip(boxes, track_ids, masks, cls, confidences):
            if class_id != 0:
                continue

            x_min, y_min, x_max, y_max = box
            mask_h, mask_w = mask.shape

            x1 = max(int(x_min * mask_w) - padding + 4, 0)
            y1 = max(int(y_min * mask_h) - padding - 2, 0)
            x2 = min(int(x_max * mask_w) + padding - 4, mask_w)
            y2 = min(int(y_max * mask_h) + padding + 4, mask_h)

            if x1 >= x2 or y1 >= y2:
                logger.debug(f"Frame {frame_number}, sperm {track_id}: Invalid region bounds")
                continue

            region = mask[y1:y2, x1:x2]
            frame_region = frame[y1:y2, x1:x2]

            if region.size == 0 or frame_region.size == 0:
                logger.debug(f"Frame {frame_number}, sperm {track_id}: Empty region/frame")
                continue

            # ── Mask refinement ──
            try:
                region_refined = refine_mask(region, mask_refine_cfg)
                quality_metrics = compute_mask_quality_metrics(region, region_refined)
                logger.debug(
                    f"Mask refinement: id={track_id}, Dice={quality_metrics['dice_coefficient']:.3f}"
                )
            except Exception as e:
                logger.warning(f"Mask refinement failed for sperm {track_id}: {e}")
                region_refined = region

            gray_region = (
                cv2.cvtColor(frame_region, cv2.COLOR_BGR2GRAY)
                if len(frame_region.shape) == 3
                else frame_region
            )

            contours, _ = cv2.findContours(region_refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                logger.debug(f"Frame {frame_number}, sperm {track_id}: No contours found")
                continue

            largest_contour = max(contours, key=cv2.contourArea)

            if track_id not in mask_info_dict:
                mask_info_dict[track_id] = {
                    "video_name": "",          # filled by caller from video_name
                    "track_id": track_id,
                    "data": [],
                    "Positions": [],
                    "frame_count": 0,
                    "current_score": -1,
                    "score_history": [],
                }

            mask_info_dict[track_id]["frame_count"] += 1
            metrics.log_detection(track_id, confidence)

            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            mask_binary = region_refined > 0
            mean_brightness = (
                np.mean(gray_region[mask_binary])
                if gray_region.shape == region_refined.shape and np.any(mask_binary)
                else -1
            )

            position_data = {
                "posX": center_x,
                "posY": center_y,
                "MeanBrightness": mean_brightness,
            }
            mask_info_dict[track_id]["Positions"].append(position_data)

            # ── Extract and validate morpho features ──
            frame_info = extract_morpho_features(largest_contour, box, region_refined)
            frame_info["frame"] = cap.get(cv2.CAP_PROP_POS_FRAMES)
            
            # Validate features
            feature_validation = validate_morpho_features(frame_info)
            if not feature_validation.is_valid:
                logger.warning(
                    f"Morpho feature validation failed for sperm {track_id}: "
                    f"{len(feature_validation.failures)} issue(s)"
                )
                for failure in feature_validation.failures:
                    metrics.log_feature_validation_failure(
                        track_id, failure["feature"], failure["reason"]
                    )
            
            mask_info_dict[track_id]["data"].append(frame_info)

            # Live score snapshot
            if morpho_params is not None and motility_params is not None:
                frame_count = mask_info_dict[track_id]["frame_count"]
                if frame_count >= score_window and frame_count % score_window == 0:
                    try:
                        snap = score_snapshot(
                            track_data=mask_info_dict[track_id],
                            window_size=score_window,
                            fps=fps,
                            morpho_params=morpho_params,
                            motility_params=motility_params,
                            filter_cfg=filter_cfg,
                            model=blastocyst_model,
                            device=device,
                            video_width=width,
                            video_height=height,
                        )
                        if snap != -1:
                            mask_info_dict[track_id]["score_history"].append(snap)
                    except Exception as e:
                        logger.error(f"Score computation failed for sperm {track_id}: {e}")

                if mask_info_dict[track_id]["score_history"]:
                    mask_info_dict[track_id]["current_score"] = np.mean(
                        mask_info_dict[track_id]["score_history"]
                    )

        # Ranking and annotation
        current_ids = set(track_ids) if len(track_ids) else set()
        top_3 = ranking_state.update(mask_info_dict, current_ids, frame_number)
        draw_ranking(annotated_frame, boxes, track_ids, cls, top_3, width, height)

        writer.write(annotated_frame)
        cv2.imshow("YOLOv8 Tracking - Top 3 Ranked", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ── Log final metrics ──
    metrics.print_summary()

    return mask_info_dict
