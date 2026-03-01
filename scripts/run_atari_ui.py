"""Launcher script for Atari UI that sets up sys.path correctly."""
import os
import sys

# Ensure project root is on sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

from demos.atari.ui.__main__ import main

main()
