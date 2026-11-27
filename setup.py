"""Keep repository-root installs compatible with the original Decisions package."""
import os
from pathlib import Path

from setuptools import setup

os.chdir(Path(__file__).resolve().parent / "decisions" / "v1")
setup()
