"""
Insurance Risk Analytics - Source Code Package
AlphaCare Insurance Solutions Analysis
"""

__version__ = "0.1.0"
__author__ = "Tsegay"

# src/__init__.py
from .data_processor import InsuranceDataProcessor
from .visualizer import InsuranceVisualizer

__all__ = ['InsuranceDataProcessor', 'InsuranceVisualizer']