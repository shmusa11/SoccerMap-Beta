# football_analyzer/main.py

import os
import warnings
from analyzer import OptimizedFootballAnalyzer


def main():
    """Main function for the robust football analysis."""
    warnings.filterwarnings("ignore")

    # --- CONFIGURATION ---
    roboflow_api_key = os.environ.get("ROBOFLOW_API_KEY")

    if not roboflow_api_key:
        raise ValueError("Set ROBOFLOW_API_KEY in your environment before running this prototype.")

    # --- SET YOUR VIDEO FILE HERE ---
    # This path should point to ANY game clip that has the numerical landmarks
    video_path = "local_clip.mp4"

    output_path = f"final_analysis_universal.mp4"
    max_frames_to_process = 400  # Set to an integer (e.g., 500) for testing, or None for the whole video

    print("=" * 60)
    print(f"STARTING UNIVERSAL FOOTBALL ANALYZER")
    print(f"Video Input: {video_path}")
    print("=" * 60)

    try:
        # Simplified analyzer initialization
        analyzer = OptimizedFootballAnalyzer(roboflow_api_key=roboflow_api_key)

        analytics = analyzer.analyze_video(
            video_path,
            max_frames=max_frames_to_process,
            output_path=output_path
        )

        if analytics:
            print(f"\nAnalysis successful! Video saved to: {output_path}")
        else:
            print("Analysis did not complete successfully.")

    except Exception as e:
        print(f"\n--- ANALYSIS FAILED ---")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()