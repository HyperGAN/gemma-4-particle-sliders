"""Install the pinned dependencies for a new Gemma source cohort."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(ROOT)], check=True, cwd=ROOT)
