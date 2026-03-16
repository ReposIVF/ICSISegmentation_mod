"""
Loads config.yaml and all runtime JSON assets (scalers, filters).
Also handles resolution-matching logic for scaler configs.
"""
import json
import numpy as np
import yaml


# ── YAML ──────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    """Load the main YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── JSON assets ───────────────────────────────────────────────────────────────

def load_scalers(cfg: dict) -> tuple[dict, dict]:
    """Load morphology and motility scaler JSON files."""
    with open(cfg["paths"]["scalers"]["morpho"], "r") as f:
        morpho_scalers = json.load(f)
    with open(cfg["paths"]["scalers"]["motility"], "r") as f:
        motility_scalers = json.load(f)
    return morpho_scalers, motility_scalers


def load_filters(cfg: dict) -> dict:
    """Load and unpack the soft/hard/morpho filter config."""
    with open(cfg["paths"]["scalers"]["soft_filters"], "r") as f:
        raw = json.load(f)

    sf = raw["soft_filters"]
    return {
        "scaling_factor": sf["scaling_factor"],
        "filters_enabled": sf.get("filters_enabled", True),
        "filters": sf["filters"],
        "static_filter_enabled": sf.get("static_filter_enabled", True),
        "static_filter": sf.get("static_filter", None),
        "hard_filters_enabled": sf.get("hard_filters_enabled", True),
        "hard_filters": sf.get("hard_filters", {}),
        "morpho_filters_enabled": sf.get("morpho_filters_enabled", True),
        "morpho_filters": sf.get("morpho_filters", {}),
    }


# ── Resolution matching ───────────────────────────────────────────────────────

def _find_closest_resolution(target_w: int, target_h: int, available_configs: list) -> tuple[str, float]:
    """Return the config key whose encoded resolution is closest to (target_w, target_h)."""
    min_distance = float("inf")
    closest_config = None

    for config in available_configs:
        try:
            if "[" in config:  # Morphology format: "('10X', '[1920.0, 1080.0]', '7%')"
                res_str = config.split("[")[1].split("]")[0]
                w, h = map(float, res_str.split(","))
            else:              # Motility format: "10x-1920_1080-7%"
                res_str = config.split("-")[1]
                w, h = map(int, res_str.split("_"))

            distance = np.sqrt((w - target_w) ** 2 + (h - target_h) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_config = config
        except Exception:
            continue

    return closest_config, min_distance


def resolve_scaler_params(
    morpho_scalers: dict,
    motility_scalers: dict,
    width: int,
    height: int,
    magnification: str,
    solution_pct: str,
) -> tuple[dict | None, dict | None]:
    """
    Given video resolution + user choices, return the matching morpho and motility
    scaler parameter dicts. Falls back to closest resolution if exact match not found.
    """
    morpho_config = f"('{magnification}', '[{float(width)}, {float(height)}]', '{solution_pct}')"
    motility_config = f"{magnification.lower()}-{width}_{height}-{solution_pct}"

    # ── Morphology ────────────────────────────────────────────────────────────
    if morpho_config in morpho_scalers:
        print(f"Using exact morphology config: {morpho_config}")
        morpho_params = morpho_scalers[morpho_config]
    else:
        print(f"Exact morphology config not found: {morpho_config}")
        filtered = [c for c in morpho_scalers if magnification in c and solution_pct in c]
        if filtered:
            closest, distance = _find_closest_resolution(width, height, filtered)
            print(f"Using closest morphology config: {closest} (distance: {distance:.1f} px)")
            morpho_params = morpho_scalers[closest]
        else:
            print("Warning: No compatible morphology config found!")
            morpho_params = None

    # ── Motility ──────────────────────────────────────────────────────────────
    if motility_config in motility_scalers:
        print(f"Using exact motility config: {motility_config}")
        motility_params = motility_scalers[motility_config]
    else:
        print(f"Exact motility config not found: {motility_config}")
        filtered = [c for c in motility_scalers if magnification.lower() in c and solution_pct in c]
        if filtered:
            closest, distance = _find_closest_resolution(width, height, filtered)
            print(f"Using closest motility config: {closest} (distance: {distance:.1f} px)")
            motility_params = motility_scalers[closest]
        else:
            print("Warning: No compatible motility config found!")
            motility_params = None

    return morpho_params, motility_params
