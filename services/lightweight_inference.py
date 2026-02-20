"""
Lightweight image analysis using OpenCV edge detection.
Uses edge density as a proxy for surface damage (cracks/potholes).
"""

import cv2
import numpy as np


def analyze_image(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        return {"severity": "low", "confidence_score": 0.0}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size

    confidence = float(min(edge_density * 5, 1.0))

    if confidence < 0.3:
        severity = "low"
    elif confidence < 0.6:
        severity = "medium"
    else:
        severity = "high"

    return {
        "severity": severity,
        "confidence_score": round(confidence, 2),
    }
