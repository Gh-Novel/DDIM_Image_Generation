"""Pytest config: make the repo root importable so tests can do
`from models.unet import UNet` exactly like the training scripts do.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
