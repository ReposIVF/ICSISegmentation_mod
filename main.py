"""
main.py — Entry point for the sperm tracking and scoring pipeline.

Batch mode: all video files found in config.paths.videos_dir are processed
sequentially. Prompts for clinic, magnification, and solution percentage are
asked once per video so different setups can coexist in the same folder.

Single-video mode:
    python main.py --video /path/to/video.mp4

Batch mode:
    python main.py
"""
import os
import argparse
import cv2

from data_io.config_loader import load_config, load_scalers, load_filters, resolve_scaler_params
from data_io.video_io import open_video, create_writer
from data_io.csv_exporter import export_csv
from models.model_loader import load_yolo, load_tabtransformer
from core.tracker import run_tracking_loop
from core.scorer import post_process
from utils.device import get_device
from utils.logger import setup_logger, get_logger


# ── Helpers ───────────────────────────────────────────────────────────────────

def prompt_choice(prompt_text: str, options: list) -> str:
    """Display a numbered menu and return the selected option."""
    print(f"\n{prompt_text}")
    for i, opt in enumerate(options, start=1):
        print(f"  {i}. {opt}")
    while True:
        try:
            choice = int(input("Enter number: "))
            if 1 <= choice <= len(options):
                return options[choice - 1]
            print(f"  Please choose a number between 1 and {len(options)}.")
        except ValueError:
            print("  Please enter a valid number.")


def discover_videos(videos_dir: str) -> list[str]:
    """Return sorted list of .mp4 / .avi / .mov files in videos_dir."""
    supported = {".mp4", ".avi", ".mov", ".mkv"}
    paths = [
        os.path.join(videos_dir, f)
        for f in sorted(os.listdir(videos_dir))
        if os.path.splitext(f)[1].lower() in supported
    ]
    return paths


# ── Pipeline for a single video ───────────────────────────────────────────────

def process_video(
    video_path: str,
    cfg: dict,
    morpho_scalers: dict,
    motility_scalers: dict,
    filter_cfg: dict,
    yolo_model,
    blastocyst_model,
    device: str,
) -> None:
    logger = get_logger()
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    logger.info("=" * 70)
    logger.info(f"Processing video: {video_name}")
    logger.info("=" * 70)

    # ── User prompts ──────────────────────────────────────────────
    opts = cfg["prompt_options"]
    clinic       = prompt_choice("Select clinic:",              opts["clinics"])
    magnification= prompt_choice("Select magnification:",       opts["magnifications"])
    solution_pct = prompt_choice("Select solution percentage:", opts["solution_percentages"])

    logger.info(f"Video settings - Clinic: {clinic}, Magnification: {magnification}, Solution: {solution_pct}")

    # ── Video I/O ─────────────────────────────────────────────────
    cap, width, height, fps = open_video(video_path)
    logger.info(f"Video opened: {width}x{height} @ {fps} fps")

    writer = create_writer(
        video_path,
        cfg["paths"]["videos_results_dir"],
        width, height, fps,
    )

    # ── Resolve scaler params for this video's resolution + settings ──
    morpho_params, motility_params = resolve_scaler_params(
        morpho_scalers, motility_scalers,
        width, height, magnification, solution_pct,
    )
    logger.info(f"Scaler parameters resolved for {magnification}@{solution_pct}")

    # ── Tracking loop ─────────────────────────────────────────────
    logger.info("Starting tracking loop...")
    mask_info_dict = run_tracking_loop(
        cap=cap,
        writer=writer,
        yolo_model=yolo_model,
        blastocyst_model=blastocyst_model,
        device=device,
        fps=fps,
        width=width,
        height=height,
        morpho_params=morpho_params,
        motility_params=motility_params,
        filter_cfg=filter_cfg,
        cfg=cfg,
    )
    logger.info(f"Tracking loop completed. Tracked {len(mask_info_dict)} sperm")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # Attach video_name to each tracked sperm
    for data in mask_info_dict.values():
        data["video_name"] = video_name

    # ── Post-processing & CSV export ──────────────────────────────
    logger.info("Running post-processing...")
    results = post_process(
        mask_info_dict=mask_info_dict,
        fps=fps,
        morpho_params=morpho_params,
        motility_params=motility_params,
        filter_cfg=filter_cfg,
        model=blastocyst_model,
        device=device,
        video_width=width,
        video_height=height,
        video_name=video_name,
    )

    export_csv(results, clinic, cfg["paths"]["data_results_dir"], video_name)
    logger.info(f"CSV export completed: {cfg['paths']['data_results_dir']}/{video_name}.csv")
    logger.info(f"✅ Completed: {video_name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sperm tracking and scoring pipeline.")
    parser.add_argument("--video", type=str, default=None, help="Path to a specific video file to process.")
    args = parser.parse_args()

    # ── Initialize logging ──
    setup_logger(log_dir="./logs")
    logger = get_logger()
    logger.info("=" * 70)
    logger.info("SPERM TRACKING AND SCORING PIPELINE - STARTING")
    logger.info("=" * 70)

    # Load config and shared assets (done once for all videos)
    cfg = load_config("config.yaml")
    logger.info(f"Configuration loaded from config.yaml")
    device = get_device()
    logger.info(f"Using device: {device}")
    morpho_scalers, motility_scalers = load_scalers(cfg)
    logger.info(f"Morphology and motility scalers loaded")
    filter_cfg = load_filters(cfg)
    logger.info(f"Filter configuration loaded")

    # Create output directories if needed
    os.makedirs(cfg["paths"]["videos_results_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["data_results_dir"], exist_ok=True)
    logger.info(f"Output directories ready: {cfg['paths']['videos_results_dir']}, {cfg['paths']['data_results_dir']}")

    # Load models (done once, shared across all videos)
    logger.info("Loading YOLO model...")
    yolo_model       = load_yolo(cfg["paths"]["yolo_model"])
    logger.info("Loading TabTransformer model...")
    blastocyst_model = load_tabtransformer(cfg["paths"]["tabtransformer_model"], cfg, device)
    logger.info("All models loaded successfully")

    if args.video:
        if not os.path.isfile(args.video):
            logger.error(f"Video file not found: {args.video}")
            return
        video_paths = [args.video]
        logger.info(f"Processing single video: {args.video}")
    else:
        video_paths = discover_videos(cfg["paths"]["videos_dir"])
        if not video_paths:
            logger.warning(f"No videos found in: {cfg['paths']['videos_dir']}")
            return
        logger.info(f"Found {len(video_paths)} video(s) to process")

    for video_path in video_paths:
        process_video(
            video_path=video_path,
            cfg=cfg,
            morpho_scalers=morpho_scalers,
            motility_scalers=motility_scalers,
            filter_cfg=filter_cfg,
            yolo_model=yolo_model,
            blastocyst_model=blastocyst_model,
            device=device,
        )

    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)
    logger.info(f"Log file saved to: ./logs/sperm_pipeline.log")


if __name__ == "__main__":
    main()
