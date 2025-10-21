#!/usr/bin/env python3
"""
Evaluation Analysis (QF0 experiment)

Wrapper that sets OCO3_EXP_SUFFIX=QF0 and reuses evaluation_analysis_plots to
generate confusion matrix and summary stats, including precision/recall.
"""

import os
import sys

os.environ.setdefault('OCO3_EXP_SUFFIX', 'QF0')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.analysis.evaluation_analysis_plots import main as base_main


if __name__ == "__main__":
    sys.exit(base_main())


