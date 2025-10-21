#!/usr/bin/env python3
"""
ROC Curve Generation for OCO-3 Swath Bias Correction (QF0 experiment)

Wrapper around classifier_performance_curves that switches to the QF0 experiment
via environment suffix to load the correct CV models and results, and then
generates precision/recall metrics on independent (CV) data.
"""

import os
import sys

# Ensure we point to the QF0 experiment
os.environ.setdefault('OCO3_EXP_SUFFIX', 'QF0')

# Reuse the existing script's main
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.analysis.classifier_performance_curves import main as base_main


if __name__ == "__main__":
    sys.exit(base_main())


