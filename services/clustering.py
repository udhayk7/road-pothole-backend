"""
Clustering service using DBSCAN algorithm for geographic report grouping.
Groups nearby reports into clusters based on their latitude/longitude coordinates.
"""

import numpy as np
from sklearn.cluster import DBSCAN
from sqlalchemy.orm import Session
from typing import Dict, List

from models import Report, Cluster
from services.priority import (
    calculate_priority_score, 
    get_dominant_road_type, 
    get_road_type_weight,
    calculate_cluster_centroid
)
from services.ward_lookup import get_ward_name


# Severity level to numeric mapping
SEVERITY_MAP: Dict[str, int] = {
    "low": 1,
    "medium": 2,
    "high": 3
}

# DBSCAN parameters
# eps ≈ 0.0001 corresponds to roughly 11 meters at the equator
DBSCAN_EPS = 0.0001
DBSCAN_MIN_SAMPLES = 1


def _contractor_for_ward(ward_name: str) -> str:
    """Assign contractor_name based on ward_name."""
    if not ward_name:
        return "PWD Kerala Division"
    ward = ward_name.strip()
    if "Kochi" in ward:
        return "PWD Kochi Division"
    if "Trivandrum" in ward:
        return "PWD South Zone"
    return "PWD Kerala Division"


def severity_to_numeric(severity: str) -> int:
    """
    Convert severity string to numeric value.
    
    Args:
        severity: Severity level string (low, medium, high)
        
    Returns:
        Numeric severity value (1, 2, or 3)
    """
    return SEVERITY_MAP.get(severity.lower(), 1)


def run_clustering(db: Session) -> None:
    """
    Execute DBSCAN clustering on all reports and update cluster assignments.
    """

    # 1️⃣ Fetch all reports
    reports = db.query(Report).all()

    # 2️⃣ Always clear old cluster references
    db.query(Report).update({Report.cluster_id: None})
    db.commit()

    # 3️⃣ Always delete old clusters
    db.query(Cluster).delete()
    db.commit()

    # 4️⃣ If not enough reports, stop after cleanup
    if len(reports) < DBSCAN_MIN_SAMPLES:
        return

    # Prepare coordinate data for clustering
    coordinates = np.array([[r.latitude, r.longitude] for r in reports])

    # Run DBSCAN
    clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    cluster_labels = clustering.fit_predict(coordinates)

    # Get unique cluster labels (exclude noise = -1)
    unique_labels = set(cluster_labels)
    unique_labels.discard(-1)

    cluster_id_mapping: Dict[int, int] = {}

    # 5️⃣ Create clusters
    for label in unique_labels:
        cluster_indices = np.where(cluster_labels == label)[0]
        cluster_reports = [reports[i] for i in cluster_indices]

        severities = [severity_to_numeric(r.severity) for r in cluster_reports]
        avg_severity = sum(severities) / len(severities)
        report_count = len(cluster_reports)

        dominant_road_type = get_dominant_road_type(cluster_reports)
        road_type_weight = get_road_type_weight(dominant_road_type)

        center_lat, center_lon = calculate_cluster_centroid(cluster_reports)
        ward_name = get_ward_name(center_lat, center_lon)

        # Higher rain factor for high-rainfall wards (Piravom, Kochi, Ernakulam)
        ward_upper = (ward_name or "").strip().upper()
        rain_factor = 1.3 if (
            "PIRAVOM" in ward_upper or "KOCHI" in ward_upper or "ERNAKULAM" in ward_upper
        ) else 1.0

        priority_score = calculate_priority_score(
            avg_severity,
            report_count,
            road_type_weight,
            rain_factor
        )

        contractor_name = _contractor_for_ward(ward_name)

        # Predictive and financial calculations (do not affect clustering or priority)
        predicted_failure_days = max(3, int(30 / (avg_severity * rain_factor))) if (avg_severity * rain_factor) > 0 else 30
        estimated_repair_cost = report_count * 12000 * road_type_weight
        delayed_repair_cost = estimated_repair_cost * 1.8
        cost_savings = delayed_repair_cost - estimated_repair_cost
        if priority_score > 6:
            risk_category = "Critical"
        elif priority_score > 4:
            risk_category = "High"
        elif priority_score > 2:
            risk_category = "Medium"
        else:
            risk_category = "Low"

        new_cluster = Cluster(
            avg_severity=round(avg_severity, 2),
            report_count=report_count,
            priority_score=round(priority_score, 2),
            dominant_road_type=dominant_road_type,
            road_type_weight=road_type_weight,
            rain_factor=rain_factor,
            ward_name=ward_name,
            center_lat=round(center_lat, 6),
            center_lon=round(center_lon, 6),
            status="pending",
            contractor_name=contractor_name,
            predicted_failure_days=predicted_failure_days,
            estimated_repair_cost=round(estimated_repair_cost),
            delayed_repair_cost=round(delayed_repair_cost),
            cost_savings=round(cost_savings),
            risk_category=risk_category,
        )

        db.add(new_cluster)
        db.flush()  # Get ID without committing
        cluster_id_mapping[label] = new_cluster.id

    # 6️⃣ Assign cluster_id to reports
    for i, report in enumerate(reports):
        label = cluster_labels[i]
        if label != -1:
            report.cluster_id = cluster_id_mapping[label]

    # 7️⃣ Commit ONCE after all assignments
    db.commit()


async def update_cluster_weather(db: Session, cluster_id: int) -> None:
    """
    Update a cluster's rain factor based on current weather.
    
    Args:
        db: SQLAlchemy database session
        cluster_id: ID of the cluster to update
    """
    from services.weather import get_rain_factor
    
    cluster = db.query(Cluster).filter(Cluster.id == cluster_id).first()
    
    if not cluster or not cluster.center_lat or not cluster.center_lon:
        return
    
    # Fetch rain factor from Open-Meteo weather API
    rain_factor = await get_rain_factor(cluster.center_lat, cluster.center_lon)
    
    # Update cluster with new rain factor and recalculate priority
    cluster.rain_factor = rain_factor
    cluster.priority_score = round(
        calculate_priority_score(
            cluster.avg_severity,
            cluster.report_count,
            cluster.road_type_weight,
            rain_factor
        ),
        2
    )
    
    db.commit()


async def update_all_clusters_weather(db: Session) -> None:
    """
    Update weather data for all clusters.
    
    Args:
        db: SQLAlchemy database session
    """
    from services.weather import get_rain_factor
    
    clusters = db.query(Cluster).all()
    
    for cluster in clusters:
        if cluster.center_lat and cluster.center_lon:
            rain_factor = await get_rain_factor(cluster.center_lat, cluster.center_lon)
            
            cluster.rain_factor = rain_factor
            cluster.priority_score = round(
                calculate_priority_score(
                    cluster.avg_severity,
                    cluster.report_count,
                    cluster.road_type_weight,
                    rain_factor
                ),
                2
            )
    
    db.commit()


def get_cluster_statistics(db: Session, cluster_id: int) -> Dict:
    """
    Get detailed statistics for a specific cluster.
    
    Args:
        db: SQLAlchemy database session
        cluster_id: ID of the cluster
        
    Returns:
        Dictionary containing cluster statistics
    """
    cluster = db.query(Cluster).filter(Cluster.id == cluster_id).first()
    
    if not cluster:
        return {}
    
    reports = db.query(Report).filter(Report.cluster_id == cluster_id).all()
    
    return {
        "cluster_id": cluster.id,
        "avg_severity": cluster.avg_severity,
        "report_count": cluster.report_count,
        "priority_score": cluster.priority_score,
        "status": cluster.status,
        "reports": [
            {
                "id": r.id,
                "latitude": r.latitude,
                "longitude": r.longitude,
                "severity": r.severity
            }
            for r in reports
        ]
    }
