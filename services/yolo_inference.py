"""
YOLOv8 local inference for road damage detection.
Uses ultralytics YOLO model (yolov8n.pt) for object detection.
"""

from typing import Dict, Any

from ultralytics import YOLO

# Load model globally
model = YOLO("yolov8n.pt")


def _confidence_to_severity(confidence: float) -> str:
    """
    Map confidence score to severity.
    conf < 0.4 → low
    0.4–0.7 → medium
    >= 0.7 → high
    """
    if confidence < 0.4:
        return "low"
    if confidence < 0.7:
        return "medium"
    return "high"


def analyze_image(image_path: str) -> Dict[str, Any]:
    """
    Run YOLOv8 inference on an image file.

    Args:
        image_path: Local path to the image file.

    Returns:
        Dict with "severity" (str) and "confidence_score" (float).
        severity is one of: "none", "low", "medium", "high".
        If no detections: {"severity": "none", "confidence_score": 0.0}.
    """
    results = model(image_path)

    if not results or len(results) == 0:
        return {"severity": "none", "confidence_score": 0.0}

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return {"severity": "none", "confidence_score": 0.0}

    # Extract confidence from first detection
    conf = boxes.conf
    if conf is None or len(conf) == 0:
        return {"severity": "none", "confidence_score": 0.0}

    confidence = float(conf[0].item())
    severity = _confidence_to_severity(confidence)

    return {
        "severity": severity,
        "confidence_score": round(confidence, 2),
    }
