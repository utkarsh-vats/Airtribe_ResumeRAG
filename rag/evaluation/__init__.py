"""Evaluation against the ground truth (``data/ground_truth``).

This package is the only part of ``rag`` that reads labels (``qrels.json``, ``manifest.json``).
Retrieval, scoring, calibration and matching never import it and never open those files.
"""
