"""
Stable real-time ranking of the top-3 sperm visible in the current frame.

Uses:
  - EMA smoothing to dampen noisy score updates
  - Periodic update interval so the display doesn't flicker every frame
  - Hysteresis threshold so minor score changes don't reshuffle the ranking
"""
import cv2
from core.scorer import normalize_scores_dynamic


# Visual config for rank badges
_RANK_STYLE = {
    0: {"color": (0, 255, 0),   "label": "1st"},   # Green
    1: {"color": (0, 255, 255), "label": "2nd"},   # Yellow
    2: {"color": (0, 0, 255),   "label": "3rd"},   # Red
}


class RankingState:
    """Holds all mutable state for the stable top-3 ranking across frames."""

    def __init__(self, update_interval: int, change_threshold: float, smoothing_alpha: float):
        self.update_interval = update_interval
        self.change_threshold = change_threshold
        self.smoothing_alpha = smoothing_alpha

        self.smoothed_scores: dict[int, float] = {}
        self.frozen_top_3: list[tuple[int, float]] = []
        self.last_update_frame: int = 0

    def update(self, mask_info_dict: dict, current_track_ids: set, frame_number: int) -> list[tuple[int, float]]:
        """
        Update EMA-smoothed scores for visible sperm, apply hysteresis, and
        return the stable frozen top-3 list.

        Args:
            mask_info_dict    : full tracking dict (track_id → sperm data)
            current_track_ids : set of track IDs visible in this frame
            frame_number      : current frame index

        Returns:
            List of up to 3 (track_id, smoothed_score) tuples.
        """
        # Gather raw scores for all tracked sperm (normalize globally, not just visible)
        raw_scores = {
            tid: data["current_score"]
            for tid, data in mask_info_dict.items()
            if data["current_score"] > 0
        }

        # Normalize relative to visible sample
        normalized = normalize_scores_dynamic(raw_scores)

        # EMA smoothing
        for tid, norm_score in normalized.items():
            if tid in self.smoothed_scores:
                self.smoothed_scores[tid] = (
                    self.smoothing_alpha * norm_score
                    + (1 - self.smoothing_alpha) * self.smoothed_scores[tid]
                )
            else:
                self.smoothed_scores[tid] = norm_score

        # Drop sperm no longer visible
        self.smoothed_scores = {
            tid: s for tid, s in self.smoothed_scores.items() if tid in current_track_ids
        }

        # Build candidate top-3 from current smoothed scores
        scored = sorted(self.smoothed_scores.items(), key=lambda x: x[1], reverse=True)
        candidate_top_3 = scored[:3]

        # Decide whether to update the frozen ranking
        frames_since_update = frame_number - self.last_update_frame
        if frames_since_update >= self.update_interval or not self.frozen_top_3:
            if self.frozen_top_3:
                frozen_ids = {tid for tid, _ in self.frozen_top_3}
                candidate_ids = {tid for tid, _ in candidate_top_3}
                if frozen_ids == candidate_ids:
                    # Same IDs: only update if score change is large enough
                    max_diff = max(
                        abs(new - next((s for t, s in self.frozen_top_3 if t == tid), 0))
                        for tid, new in candidate_top_3
                    )
                    if max_diff > self.change_threshold:
                        self.frozen_top_3 = candidate_top_3
                        self.last_update_frame = frame_number
                else:
                    self.frozen_top_3 = candidate_top_3
                    self.last_update_frame = frame_number
            else:
                self.frozen_top_3 = candidate_top_3
                self.last_update_frame = frame_number

        # Remove stale IDs (left the frame) from frozen list
        self.frozen_top_3 = [(tid, s) for tid, s in self.frozen_top_3 if tid in current_track_ids]

        return self.frozen_top_3


def draw_ranking(
    frame,
    boxes,
    track_ids,
    cls,
    top_3: list[tuple[int, float]],
    video_width: int,
    video_height: int,
) -> None:
    """
    Draw colored circles and rank labels on top-3 sperm in the annotated frame.
    Modifies frame in-place.
    """
    top_3_map = {tid: (rank, score) for rank, (tid, score) in enumerate(top_3)}

    for box, track_id, class_id in zip(boxes, track_ids, cls):
        if class_id != 0 or track_id not in top_3_map:
            continue

        rank, norm_score = top_3_map[track_id]
        style = _RANK_STYLE[rank]
        color = style["color"]
        label = style["label"]

        x_min, y_min, x_max, y_max = box
        cx = int((x_min + x_max) / 2 * video_width)
        cy = int((y_min + y_max) / 2 * video_height)
        radius = int(max((x_max - x_min) * video_width, (y_max - y_min) * video_height) / 2 * 1.2)

        cv2.circle(frame, (cx, cy), radius, color, 2)
        text = f"{label} ID:{track_id} | {norm_score:.3f}"
        cv2.putText(frame, text, (cx - 60, cy - radius - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
