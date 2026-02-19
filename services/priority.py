"""
Priority calculation service for road infrastructure clusters.
Determines repair priority based on severity, report density, road type, and weather.
Optimized for Kerala road infrastructure with monsoon-aware scoring.
"""

from sqlalchemy.orm import Session
from typing import List, Dict, Tuple
from collections import Counter

from models import Cluster, Report


# Road type weights for Kerala roads
# Higher weights for major roads that carry more traffic
ROAD_TYPE_WEIGHTS: Dict[str, float] = {
    # Major roads - highest priority
    "primary": 2.0,
    "primary_link": 2.0,
    "trunk": 2.0,
    "trunk_link": 2.0,
    
    # Secondary roads - high priority
    "secondary": 1.5,
    "secondary_link": 1.5,
    
    # Tertiary roads - medium-high priority
    "tertiary": 1.3,
    "tertiary_link": 1.3,
    
    # Local roads - standard priority
    "residential": 1.0,
    "living_street": 1.0,
    "unclassified": 1.0,
    
    # Service roads
    "service": 0.8,
    
    # Pedestrian/cycling paths - lower priority for vehicle damage
    "pedestrian": 0.7,
    "footway": 0.6,
    "cycleway": 0.7,
    "path": 0.5,
    
    # Unknown - default weight
    "unknown": 1.0,
}


def get_road_type_weight(road_type: str) -> float:
    """
    Get the priority weight for a given road type.
    
    Args:
        road_type: OSM highway type string
        
    Returns:
        Weight multiplier for priority calculation
    """
    if not road_type:
        return ROAD_TYPE_WEIGHTS["unknown"]
    return ROAD_TYPE_WEIGHTS.get(road_type.lower(), 1.0)


def get_dominant_road_type(reports: List[Report]) -> str:
    """
    Determine the dominant (most common) road type in a cluster.
    If there's a tie, prefer higher-weight road types.
    
    Args:
        reports: List of Report objects in the cluster
        
    Returns:
        The dominant road type string
    """
    if not reports:
        return "unknown"
    
    road_types = [r.road_type or "unknown" for r in reports]
    type_counts = Counter(road_types)
    
    # Get the most common road type(s)
    max_count = max(type_counts.values())
    most_common = [rt for rt, count in type_counts.items() if count == max_count]
    
    # If tie, return the one with highest weight
    if len(most_common) > 1:
        return max(most_common, key=lambda rt: get_road_type_weight(rt))
    
    return most_common[0]


def calculate_cluster_centroid(reports: List[Report]) -> Tuple[float, float]:
    """
    Calculate the geographic centroid of a cluster.
    
    Args:
        reports: List of Report objects in the cluster
        
    Returns:
        Tuple of (latitude, longitude) for the centroid
    """
    if not reports:
        return (0.0, 0.0)
    
    avg_lat = sum(r.latitude for r in reports) / len(reports)
    avg_lon = sum(r.longitude for r in reports) / len(reports)
    
    return (avg_lat, avg_lon)


def calculate_priority_score(
    avg_severity: float, 
    report_count: int, 
    road_type_weight: float = 1.0,
    rain_factor: float = 1.0
) -> float:
    """
    Calculate the priority score for a cluster.
    
    Formula: priority_score = avg_severity * report_count * road_type_weight * rain_factor
    
    This gives higher priority to:
    - Clusters with more severe damage (avg_severity)
    - Clusters with more reports (report_count)
    - Clusters on major roads (road_type_weight)
    - Clusters in areas with recent heavy rainfall (rain_factor)
    
    Args:
        avg_severity: Average severity of reports in the cluster (1-3 scale)
        report_count: Number of reports in the cluster
        road_type_weight: Weight based on road type (default 1.0)
        rain_factor: Weather-based multiplier (1.0 normal, 1.5 heavy rain)
        
    Returns:
        Calculated priority score
    """
    return avg_severity * report_count * road_type_weight * rain_factor


def calculate_cluster_priority_sync(
    avg_severity: float,
    report_count: int,
    road_type_weight: float,
    rain_factor: float
) -> float:
    """
    Synchronous priority calculation for use in clustering.
    
    Args:
        avg_severity: Average severity score
        report_count: Number of reports
        road_type_weight: Road type multiplier
        rain_factor: Weather multiplier
        
    Returns:
        Calculated priority score
    """
    return calculate_priority_score(
        avg_severity,
        report_count,
        road_type_weight,
        rain_factor
    )


def get_clusters_by_priority(db: Session) -> List[Cluster]:
    """
    Retrieve all clusters ordered by priority score (highest first).
    
    Args:
        db: SQLAlchemy database session
        
    Returns:
        List of Cluster objects sorted by priority_score descending
    """
    return db.query(Cluster).order_by(Cluster.priority_score.desc()).all()


def recalculate_cluster_priority(db: Session, cluster_id: int, rain_factor: float = 1.0) -> None:
    """
    Recalculate and update the priority score for a specific cluster.
    
    Args:
        db: SQLAlchemy database session
        cluster_id: ID of the cluster to update
        rain_factor: Weather-based multiplier
    """
    cluster = db.query(Cluster).filter(Cluster.id == cluster_id).first()
    
    if cluster:
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


def recalculate_all_priorities(db: Session, default_rain_factor: float = 1.0) -> None:
    """
    Recalculate priority scores for all clusters.
    
    Args:
        db: SQLAlchemy database session
        default_rain_factor: Default rain factor to use if not fetching weather
    """
    clusters = db.query(Cluster).all()
    
    for cluster in clusters:
        # Use existing rain_factor or default
        rain_factor = cluster.rain_factor if cluster.rain_factor else default_rain_factor
        
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
