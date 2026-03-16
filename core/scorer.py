"""
All scoring logic:
  - Unified VSL / VCL / HMP computation
  - Feature standardization (Yeo-Johnson + Z-score)
  - Hard filters, static-sperm filter, soft filters, morpho penalties
  - TabTransformer inference
  - Live score snapshot (used during tracking loop)
  - Full post-processing (used after the loop for CSV)
  - Dynamic score normalization
"""
import numpy as np
import torch
import scipy.signal
import scipy.fftpack
from scipy.spatial import distance as dist

from transform.power_transform_custom import yeo_johnson_transform, standardize


# ── Velocity metrics ──────────────────────────────────────────────────────────

def compute_vsl(positions: list, fps: int) -> float:
    """Straight-Line Velocity: displacement from first to last position."""
    if len(positions) < 2:
        return 0.0
    x_i, y_i = positions[0]["posX"], positions[0]["posY"]
    x_f, y_f = positions[-1]["posX"], positions[-1]["posY"]
    return dist.euclidean((x_i, y_i), (x_f, y_f)) / len(positions) * fps


def compute_vcl(positions: list, fps: int, jumps: int = 1) -> float:
    """Curvilinear Velocity: mean frame-to-frame displacement."""
    n = len(positions)
    if n < 2:
        return 0.0
    distances = [
        dist.euclidean(
            (positions[step - jumps]["posX"], positions[step - jumps]["posY"]),
            (positions[step]["posX"],         positions[step]["posY"]),
        )
        for step in range(jumps, n, jumps)
    ]
    return np.mean(distances) * fps if distances else 0.0


def compute_hmp(positions: list, fps: int) -> float:
    """Head Motion Parameter: dominant oscillation frequency from brightness signal."""
    brightness_values = [p["MeanBrightness"] for p in positions]
    signal = np.array([v for v in brightness_values if v != -1])

    if len(signal) < 30:
        return 0.0

    try:
        detrended = scipy.signal.detrend(signal)
        sample_freq = scipy.fftpack.fftfreq(detrended.size, 1 / fps)
        n_valid = len(np.where(sample_freq > 0)[0])

        if n_valid <= 2:
            return 0.0

        b, a = scipy.signal.butter(1, [0.010, 0.05], "bandpass")
        filtered = scipy.signal.lfilter(b, a, detrended)
        peaks, _ = scipy.signal.find_peaks(filtered, distance=10)

        if len(peaks) == 0:
            return 0.0

        return np.mean(filtered[peaks]) * fps
    except Exception:
        return 0.0


# ── Filter helpers ────────────────────────────────────────────────────────────

def check_hard_filters(sta_features: dict, hard_filters_cfg: dict) -> bool:
    """Return True if the sperm should be discarded by any active hard filter."""
    for _, config in hard_filters_cfg.items():
        if not config.get("enabled", True):
            continue
        feature = config["feature"]
        if feature not in sta_features:
            continue
        value = sta_features[feature]
        direction = config["direction"]
        threshold = config["threshold"]
        if direction == "below" and value < threshold:
            return True
        if direction == "above" and value > threshold:
            return True
    return False


def check_static_sperm(positions: list, static_cfg: dict, video_width: int, video_height: int) -> bool:
    """
    Return True if average pixel displacement is below threshold (sperm not moving).
    Positions are normalized [0-1]; converted to pixels for comparison.
    """
    if static_cfg is None or static_cfg.get("action", "penalize") != "discard":
        return False

    min_frames = static_cfg["min_frames"]
    threshold_px = static_cfg["displacement_threshold"]

    if len(positions) < min_frames:
        return False

    recent = positions[-min_frames:]
    total = 0.0
    for i in range(1, len(recent)):
        dx = (recent[i]["posX"] - recent[i - 1]["posX"]) * video_width
        dy = (recent[i]["posY"] - recent[i - 1]["posY"]) * video_height
        total += np.sqrt(dx ** 2 + dy ** 2)

    avg_px = total / (len(recent) - 1)
    return avg_px < threshold_px


def compute_soft_penalty(sta_features: dict, filters: dict, scaling_factor: float) -> float:
    """Sum weighted penalties for features below their Q1 thresholds."""
    total = 0.0
    for feature_name, feature_value in sta_features.items():
        if feature_name not in filters:
            continue
        cfg = filters[feature_name]
        if not cfg.get("enabled", True):
            continue
        if feature_value < cfg["threshold_q1"]:
            total += abs(cfg["weight"]) * abs(feature_value - cfg["threshold_q1"])
    return total * scaling_factor


def compute_morpho_penalty(sta_features: dict, morpho_filters_cfg: dict, scaling_factor: float) -> float:
    """Gradual penalty for morphological anomalies (large masks, flat shapes)."""
    total = 0.0
    for _, config in morpho_filters_cfg.items():
        if not config.get("enabled", True):
            continue
        feature = config["feature"]
        if feature not in sta_features:
            continue
        value = sta_features[feature]
        direction = config["direction"]
        threshold = config["threshold"]
        weight = config["weight"]
        if direction == "below" and value < threshold:
            total += abs(weight) * abs(value - threshold)
        elif direction == "above" and value > threshold:
            total += abs(weight) * abs(value - threshold)
    return total * scaling_factor


# ── Feature standardization ───────────────────────────────────────────────────

def standardize_motility(vsl: float, vcl: float, hmp: float, motility_params: dict) -> tuple:
    """Apply Yeo-Johnson + Z-score to motility features. Returns (sta_vsl, sta_vcl, sta_hmp)."""
    def _sta(value, key):
        try:
            return standardize(
                yeo_johnson_transform(np.array([value]), motility_params[key]["lambdas_"])[0],
                motility_params[key]["mean_"],
                motility_params[key]["scale_"],
            )
        except Exception:
            return 0.0

    return _sta(vsl, "VSL"), _sta(vcl, "VCL"), _sta(hmp, "HMP")


def standardize_morphology(morpho_means: dict, morpho_params: dict) -> dict:
    """Apply Yeo-Johnson + Z-score to morphology feature means."""
    mapping = {
        "sta_orientated_angle_mean": ("orientated_angle_mean", "orientated_angle_mean"),
        "sta_circularity_mean":      ("circularity_mean",      "circularity_mean"),
        "sta_convexity_mean":        ("convexity_mean",        "convexity_mean"),
        "sta_compactness_mean":      ("compactness_mean",      "compactness_mean"),
        "sta_minor_axis_radius_mean":("minor_axis_radius_mean","minor_axis_radius_mean"),
    }
    result = {}
    for sta_key, (mean_key, param_key) in mapping.items():
        try:
            result[sta_key] = standardize(
                yeo_johnson_transform(
                    np.array([morpho_means[mean_key]]),
                    morpho_params[param_key]["lambda"],
                )[0],
                morpho_params[param_key]["mean"],
                morpho_params[param_key]["scale"],
            )
        except Exception:
            result[sta_key] = 0.0
    return result


# ── Model inference ───────────────────────────────────────────────────────────

def run_model(sta_vsl, sta_vcl, sta_hmp, sta_morpho: dict, model, device: str) -> float:
    """Run TabTransformer and return sigmoid score."""
    features_array = np.array([[
        sta_vsl, sta_vcl, sta_hmp,
        sta_morpho["sta_orientated_angle_mean"],
        sta_morpho["sta_circularity_mean"],
        sta_morpho["sta_convexity_mean"],
        sta_morpho["sta_compactness_mean"],
        sta_morpho["sta_minor_axis_radius_mean"],
    ]], dtype=np.float32)

    with torch.no_grad():
        tensor = torch.tensor(features_array, dtype=torch.float32).to(device)
        logits = model(torch.empty(1, 0).to(device), tensor)
        return torch.sigmoid(logits).item()


# ── Score computation ─────────────────────────────────────────────────────────

def compute_score(
    positions: list,
    morpho_means: dict,
    fps: int,
    morpho_params: dict,
    motility_params: dict,
    filter_cfg: dict,
    model,
    device: str,
    video_width: int,
    video_height: int,
) -> float:
    """
    Compute a blastocyst score for a sperm given its positions and morphology means.
    Returns -1 if the sperm is filtered out or an error occurs.

    Used by both the live snapshot scorer and the final post-processing pass.
    """
    vsl = compute_vsl(positions, fps)
    vcl = compute_vcl(positions, fps)
    hmp = compute_hmp(positions, fps)

    sta_vsl, sta_vcl, sta_hmp = standardize_motility(vsl, vcl, hmp, motility_params)
    sta_morpho = standardize_morphology(morpho_means, morpho_params)

    all_sta = {
        "sta_VSL": sta_vsl,
        "sta_VCL": sta_vcl,
        "sta_minor_axis_radius_mean": sta_morpho["sta_minor_axis_radius_mean"],
        "sta_compactness_mean":       sta_morpho["sta_compactness_mean"],
        "sta_circularity_mean":       sta_morpho["sta_circularity_mean"],
        "sta_convexity_mean":         sta_morpho["sta_convexity_mean"],
        "sta_orientated_angle_mean":  sta_morpho["sta_orientated_angle_mean"],
    }

    if filter_cfg["hard_filters_enabled"] and check_hard_filters(all_sta, filter_cfg["hard_filters"]):
        return -1

    if filter_cfg["static_filter_enabled"] and check_static_sperm(
        positions, filter_cfg["static_filter"], video_width, video_height
    ):
        return -1

    try:
        raw_score = run_model(sta_vsl, sta_vcl, sta_hmp, sta_morpho, model, device)
    except Exception:
        return -1

    soft_features = {
        "sta_orientated_angle_mean": sta_morpho["sta_orientated_angle_mean"],
        "sta_circularity_mean":      sta_morpho["sta_circularity_mean"],
        "sta_VCL":                   sta_vcl,
        "sta_VSL":                   sta_vsl,
    }

    penalty = 0.0
    if filter_cfg["filters_enabled"]:
        penalty += compute_soft_penalty(soft_features, filter_cfg["filters"], filter_cfg["scaling_factor"])
    if filter_cfg["morpho_filters_enabled"]:
        penalty += compute_morpho_penalty(all_sta, filter_cfg["morpho_filters"], filter_cfg["scaling_factor"])

    return max(0.0, min(1.0, raw_score - penalty))


# ── Live snapshot scorer (called every SCORE_WINDOW_SIZE frames) ──────────────

def score_snapshot(
    track_data: dict,
    window_size: int,
    fps: int,
    morpho_params: dict,
    motility_params: dict,
    filter_cfg: dict,
    model,
    device: str,
    video_width: int,
    video_height: int,
) -> float:
    """
    Compute an instantaneous score from the last `window_size` frames.
    Returns -1 if there is not enough data.
    """
    if len(track_data["data"]) < window_size or len(track_data["Positions"]) < window_size:
        return -1

    recent_data = track_data["data"][-window_size:]
    recent_positions = track_data["Positions"][-window_size:]

    morpho_means = {}
    for feature in ["area", "perimeter", "orientated_angle", "circularity",
                    "convexity", "compactness", "major_axis_radius", "minor_axis_radius"]:
        values = [f[feature] for f in recent_data]
        morpho_means[f"{feature}_mean"] = np.mean(values) if values else 0.0

    return compute_score(
        positions=recent_positions,
        morpho_means=morpho_means,
        fps=fps,
        morpho_params=morpho_params,
        motility_params=motility_params,
        filter_cfg=filter_cfg,
        model=model,
        device=device,
        video_width=video_width,
        video_height=video_height,
    )


# ── Post-processing (full-track, called after the loop) ───────────────────────

MORPHO_FEATURES = [
    "area", "perimeter", "aspect_ratio", "extend", "orientated_angle",
    "circularity", "hull_area", "solidity", "hull_perimeter", "convexity",
    "eccentricity", "compactness", "major_axis_radius", "minor_axis_radius",
]


def post_process(
    mask_info_dict: dict,
    fps: int,
    morpho_params: dict | None,
    motility_params: dict | None,
    filter_cfg: dict,
    model,
    device: str,
    video_width: int,
    video_height: int,
    video_name: str,
) -> list[dict]:
    """
    Compute final per-sperm scores over the full track and return a list of result dicts.
    Appends normalized_score using the full-sample dynamic normalization.
    """
    mean_dict = {}

    for track_id, info in mask_info_dict.items():
        if not info["data"]:
            continue

        # Per-feature means across all frames
        feature_means = {}
        for feature in MORPHO_FEATURES:
            values = [f[feature] for f in info["data"]]
            try:
                feature_means[f"{feature}_mean"] = np.mean(values)
            except TypeError:
                feature_means[f"{feature}_mean"] = np.nan

        vsl = compute_vsl(info["Positions"], fps)
        vcl = compute_vcl(info["Positions"], fps)
        hmp = compute_hmp(info["Positions"], fps)

        if morpho_params is not None and motility_params is not None:
            sta_vsl, sta_vcl, sta_hmp = standardize_motility(vsl, vcl, hmp, motility_params)
            sta_morpho = standardize_morphology(feature_means, morpho_params)

            blastocyst_score = compute_score(
                positions=info["Positions"],
                morpho_means=feature_means,
                fps=fps,
                morpho_params=morpho_params,
                motility_params=motility_params,
                filter_cfg=filter_cfg,
                model=model,
                device=device,
                video_width=video_width,
                video_height=video_height,
            )
        else:
            sta_vsl = sta_vcl = sta_hmp = 0.0
            sta_morpho = {k: 0.0 for k in [
                "sta_orientated_angle_mean", "sta_circularity_mean",
                "sta_convexity_mean", "sta_compactness_mean", "sta_minor_axis_radius_mean",
            ]}
            blastocyst_score = -1

        mean_dict[track_id] = {
            "video_name": video_name,
            "track_id": track_id,
            "frame_count": info["frame_count"],
            "vsl": vsl,
            "vcl": vcl,
            "hmp": hmp,
            "sta_vsl": sta_vsl,
            "sta_vcl": sta_vcl,
            "sta_hmp": sta_hmp,
            **sta_morpho,
            "blastocyst_score": blastocyst_score,
            **feature_means,
        }

    results = list(mean_dict.values())

    # Dynamic normalization across the full sample
    raw_scores = {d["track_id"]: d["blastocyst_score"] for d in results if d["blastocyst_score"] > 0}
    normalized = normalize_scores_dynamic(raw_scores)
    for d in results:
        d["normalized_score"] = normalized.get(d["track_id"], -1)

    return results


# ── Score normalization ───────────────────────────────────────────────────────

def normalize_scores_dynamic(scores_dict: dict) -> dict:
    """
    Min-Max normalization with dynamic floor/ceiling based on sample quality.

    Ceiling: higher when the best raw score is higher (good sample → reward it).
    Floor:   min 0.15; shrinks when scores are diverse (to show contrast).
    """
    valid = {tid: s for tid, s in scores_dict.items() if s > 0}
    if not valid:
        return {}

    if len(valid) == 1:
        tid, raw = next(iter(valid.items()))
        ceiling = min(0.65 + raw * 0.40, 0.98)
        return {tid: round(ceiling, 4)}

    raw_values = list(valid.values())
    raw_min, raw_max = min(raw_values), max(raw_values)
    spread = raw_max - raw_min

    ceiling = min(0.55 + raw_max * 0.55, 0.98)
    floor = max(0.15, ceiling - 0.55)

    if spread < 0.05:
        floor = max(0.10, ceiling - 0.15)

    normalized = {}
    for tid, score in valid.items():
        if raw_max == raw_min:
            normalized[tid] = round(ceiling, 4)
        else:
            ratio = (score - raw_min) / (raw_max - raw_min)
            normalized[tid] = round(floor + ratio * (ceiling - floor), 4)

    return normalized
