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
DBSCAN_MIN_SAMPLES = 2


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
    
    This function:
    1. Fetches all reports from the database
    2. Applies DBSCAN clustering based on geographic coordinates
    3. Creates/updates clusters with aggregated statistics
    4. Assigns cluster_id to each report
    
    Args:
        db: SQLAlchemy database session
    """
    # Fetch all reports
    reports = db.query(Report).all()
    
    if len(reports) < DBSCAN_MIN_SAMPLES:
        # Not enough reports to form clusters
        return
    
    # Prepare coordinate data for clustering
    coordinates = np.array([[r.latitude, r.longitude] for r in reports])
    
    # Run DBSCAN clustering
    clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    cluster_labels = clustering.fit_predict(coordinates)
    
    # Get unique cluster labels (excluding noise points labeled as -1)
    unique_labels = set(cluster_labels)
    unique_labels.discard(-1)  # Remove noise label
    
    # Clear existing cluster assignments for reports that will be reclustered
    for report in reports:
        report.cluster_id = None
    
    # Delete existing clusters to rebuild fresh
    db.query(Cluster).delete()
    db.flush()
    
    # Process each cluster
    cluster_id_mapping: Dict[int, int] = {}  # Maps DBSCAN label to DB cluster ID
    
    for label in unique_labels:
        # Get indices of reports in this cluster
        cluster_indices = np.where(cluster_labels == label)[0]
        cluster_reports = [reports[i] for i in cluster_indices]
        
        # Calculate cluster statistics
        severities = [severity_to_numeric(r.severity) for r in cluster_reports]
        avg_severity = sum(severities) / len(severities)
        report_count = len(cluster_reports)
        
        # Get dominant road type and its weight for priority calculation
        dominant_road_type = get_dominant_road_type(cluster_reports)
        road_type_weight = get_road_type_weight(dominant_road_type)
        
        # Calculate cluster centroid for weather lookups and ward assignment
        center_lat, center_lon = calculate_cluster_centroid(cluster_reports)
        
        # Get ward name for cluster centroid
        ward_name = get_ward_name(center_lat, center_lon)
        
        # Default rain factor (will be updated async after cluster creation)
        rain_factor = 1.0
        
        # Calculate priority with road type weight and rain factor
        priority_score = calculate_priority_score(
            avg_severity, report_count, road_type_weight, rain_factor
        )
        
        # Create new cluster with road type, weather, and ward info
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
            status="pending"
        )
        db.add(new_cluster)
        db.flush()  # Get the cluster ID
        
        # Store mapping for report assignment
        cluster_id_mapping[label] = new_cluster.id
    
    # Assign cluster IDs to reports
    for i, report in enumerate(reports):
        label = cluster_labels[i]
        if label != -1:  # Not a noise point
            report.cluster_id = cluster_id_mapping[label]
    
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
