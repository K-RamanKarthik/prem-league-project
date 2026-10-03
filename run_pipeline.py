"""
Pipeline Runner — runs all steps in sequence.

Usage:
    python run_pipeline.py          # Run everything
    python run_pipeline.py --skip-scrape  # Skip scraping (use existing data)
"""

import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import argparse


def run_pipeline(skip_scrape: bool = False):
    """Run all steps in the Premier League scouting pipeline."""
    if not skip_scrape:
        print("\n" + "=" * 60)
        print("STEP 1/3: DATA COLLECTION (PL 2025-26)")
        print("=" * 60)
        import scraper
        scraper.main()
    else:
        print("\n⏭️ Skipping data collection step (using existing raw CSVs)")

    # Step 2: Feature Engineering
    print("\n" + "=" * 60)
    print("STEP 2/3: FEATURE ENGINEERING")
    print("=" * 60)
    import features
    features.main()

    # Step 3: ML Model
    print("\n" + "=" * 60)
    print("STEP 3/3: ML PIPELINE (CLUSTERING + SIMILARITY)")
    print("=" * 60)
    import model
    model.main()

    print("\n" + "=" * 60)
    print("🎉 PIPELINE COMPLETE!")
    print("=" * 60)
    print("\nTo launch the dashboard:")
    print("  streamlit run app.py")
    print()


def main():
    parser = argparse.ArgumentParser(description="PL Player Scouting Pipeline")
    parser.add_argument(
        "--skip-scrape", action="store_true",
        help="Skip the scraping step (use existing raw CSVs)"
    )
    args = parser.parse_args()
    run_pipeline(skip_scrape=args.skip_scrape)


if __name__ == "__main__":
    main()
