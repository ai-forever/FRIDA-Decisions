"""Build the Decisions distribution with both V1 and V2 runtimes."""
import os
from pathlib import Path

from setuptools import setup

os.chdir(Path(__file__).resolve().parent / "decisions" / "v2" / "inference" / "serving")
setup()
