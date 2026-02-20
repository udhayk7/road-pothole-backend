"""
Lightweight image analysis using OpenCV edge detection.
Uses edge density and grayscale variance for damage detection and authenticity signals.
"""

import cv2
import numpy as np


def _grayscale_variance_score(gray: np.ndarray) -> float:
    """
    Compute a 0-1 score from grayscale variance.
    Real road images typically have moderate texture (variance in a reasonable range).
    Very low variance (flat/synthetic) or extreme variance (noise) score lower.
    """
    var = float(np.var(gray))
    # Normalize: typical road texture variance often in ~500–4000 range; map to 0–1
    # Low var -> low score; mid range -> high score; very high -> cap
    if var < 200:
        return max(0.0, var / 200.0)  # suspiciously flat
    if var > 8000:
        return max(0.0, 1.0 - (var - 8000) / 8000)  # noisy/artifact
    # Good range: linear map 200–4000 -> 0.5–1.0
    return min(1.0, 0.5 + (var - 200) / 7600)


def analyze_image(image_path: str) -> dict:
    """
    Analyze image for road damage and return severity, confidence, and authenticity components.
    Returns edge_density_score and grayscale_variance_score (0–1) for use in authenticity_score.
    """
    img = cv2.imread(image_path)
    if img is None:
        return {
            "severity": "low",
            "confidence_score": 0.0,
            "edge_density_score": 0.0,
            "grayscale_variance_score": 0.0,
        }

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size

    # Confidence from edge density (existing logic)
    confidence = float(min(edge_density * 5, 1.0))
    edge_density_score = float(min(edge_density * 5, 1.0))  # 0–1 for authenticity

    if confidence < 0.3:
        severity = "low"
    elif confidence < 0.6:
        severity = "medium"
    else:
        severity = "high"

    grayscale_variance_score = _grayscale_variance_score(gray)

    return {
        "severity": severity,
        "confidence_score": round(confidence, 2),
        "edge_density_score": round(edge_density_score, 4),
        "grayscale_variance_score": round(grayscale_variance_score, 4),
    }
