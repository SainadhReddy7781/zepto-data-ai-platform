import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def run_script(script_name):
    """Run a pipeline script and stop if it fails."""

    script_path = BASE_DIR / script_name

    print(f"\nRunning {script_name}...")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        check=False
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{script_name} failed with exit code {result.returncode}"
        )


def main():
    print("=" * 50)
    print("ZEPTO DATA PIPELINE")
    print("=" * 50)

    run_script("scraper.py")
    run_script("cleaner.py")
    run_script("database.py")

    print("\n" + "=" * 50)
    print("DATA PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 50)


if __name__ == "__main__":
    main()