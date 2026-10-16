"""Rebuild cleaned data, models, evaluation, dashboard CSVs and Tableau package."""

from pathlib import Path
import subprocess
import sys


def main():
    source = Path(__file__).resolve().parent
    commands = [
        ("validate_raw.py",), ("clean_data.py",), ("train_model.py",),
        ("evaluate_model.py",), ("export_dashboard.py",), ("build_tableau.py",),
        ("train_model.py", "--full-data"),
    ]
    for script, *args in commands:
        print(f"\n>>> {script} {' '.join(args)}", flush=True)
        subprocess.run([sys.executable, str(source / script), *args], check=True)


if __name__ == "__main__":
    main()
