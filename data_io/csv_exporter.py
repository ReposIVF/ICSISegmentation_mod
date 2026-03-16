"""
Writes the final per-sperm results to a CSV file after the tracking loop completes.
"""
import csv
import os


FIELDNAMES = [
    "clinic", "file", "sperm_id", "frame_count",
    "vsl", "vcl", "hmp",
    "sta_vsl", "sta_vcl", "sta_hmp",
    "sta_orientated_angle_mean", "sta_circularity_mean", "sta_convexity_mean",
    "sta_compactness_mean", "sta_minor_axis_radius_mean",
    "blastocyst_score", "normalized_score",
    "area_mean", "perimeter_mean", "aspect_ratio_mean", "extend_mean",
    "orientated_angle_mean", "circularity_mean", "hull_area_mean",
    "solidity_mean", "hull_perimeter_mean", "convexity_mean",
    "eccentricity_mean", "compactness_mean",
    "major_axis_radius_mean", "minor_axis_radius_mean",
]


def export_csv(mean_data_list: list[dict], clinic: str, output_dir: str, video_name: str) -> None:
    """
    Write per-sperm aggregated results to a CSV file.

    Args:
        mean_data_list : list of per-sperm result dicts (output of post_process)
        clinic         : clinic label selected by the user
        output_dir     : directory where the CSV is saved
        video_name     : base name of the source video (used for filename + 'file' column)
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{video_name}_segmentation_results.csv")

    with open(csv_path, mode="w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()

        for d in mean_data_list:
            writer.writerow({
                "clinic": clinic,
                "file": d["video_name"],
                "sperm_id": d["track_id"],
                "frame_count": d["frame_count"],
                "vsl": d["vsl"],
                "vcl": d["vcl"],
                "hmp": d["hmp"],
                "sta_vsl": d["sta_vsl"],
                "sta_vcl": d["sta_vcl"],
                "sta_hmp": d["sta_hmp"],
                "sta_orientated_angle_mean": d["sta_orientated_angle_mean"],
                "sta_circularity_mean": d["sta_circularity_mean"],
                "sta_convexity_mean": d["sta_convexity_mean"],
                "sta_compactness_mean": d["sta_compactness_mean"],
                "sta_minor_axis_radius_mean": d["sta_minor_axis_radius_mean"],
                "blastocyst_score": d["blastocyst_score"],
                "normalized_score": d["normalized_score"],
                "area_mean": d["area_mean"],
                "perimeter_mean": d["perimeter_mean"],
                "aspect_ratio_mean": d["aspect_ratio_mean"],
                "extend_mean": d["extend_mean"],
                "orientated_angle_mean": d["orientated_angle_mean"],
                "circularity_mean": d["circularity_mean"],
                "hull_area_mean": d["hull_area_mean"],
                "solidity_mean": d["solidity_mean"],
                "hull_perimeter_mean": d["hull_perimeter_mean"],
                "convexity_mean": d["convexity_mean"],
                "eccentricity_mean": d["eccentricity_mean"],
                "compactness_mean": d["compactness_mean"],
                "major_axis_radius_mean": d["major_axis_radius_mean"],
                "minor_axis_radius_mean": d["minor_axis_radius_mean"],
            })

    print(f"CSV saved: {csv_path}")
