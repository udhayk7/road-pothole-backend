"""
SQLAlchemy ORM models for the Predictive Road Infrastructure System.
Defines the database schema for Reports and Clusters.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship

from database import Base


class Cluster(Base):
    """
    Represents a geographic cluster of road infrastructure reports.
    Clusters are formed using DBSCAN algorithm based on report locations.
    Optimized for Kerala road infrastructure with weather-aware priority.
    """
    __tablename__ = "clusters"

    id = Column(Integer, primary_key=True, index=True)
    avg_severity = Column(Float, default=0.0)
    report_count = Column(Integer, default=0)
    priority_score = Column(Float, default=0.0)
    dominant_road_type = Column(String, default="unknown")  # Most common road type in cluster
    road_type_weight = Column(Float, default=1.0)  # Weight based on road importance
    rain_factor = Column(Float, default=1.0)  # Weather-based multiplier (1.0 normal, 1.5 heavy rain)
    ward_name = Column(String, default="Unknown")  # Kerala ward name
    center_lat = Column(Float, nullable=True)  # Cluster centroid latitude
    center_lon = Column(Float, nullable=True)  # Cluster centroid longitude
    status = Column(String, default="pending")  # pending, in_progress, completed

    # Relationship to access all reports in this cluster
    reports = relationship("Report", back_populates="cluster")

    def __repr__(self):
        return f"<Cluster(id={self.id}, ward={self.ward_name}, priority_score={self.priority_score})>"


class Report(Base):
    """
    Represents a single road infrastructure report (e.g., pothole, crack).
    Each report contains location data, severity assessment, and optional image.
    """
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    severity = Column(String, nullable=False)  # low, medium, high
    image_path = Column(String, nullable=True)
    road_name = Column(String, nullable=True, default="Unknown")
    road_type = Column(String, nullable=True, default="unknown")
    ward_name = Column(String, nullable=True, default="Unknown")  # Kerala ward name
    created_at = Column(DateTime, default=datetime.utcnow)
    cluster_id = Column(Integer, ForeignKey("clusters.id"), nullable=True)

    # Relationship to access the parent cluster
    cluster = relationship("Cluster", back_populates="reports")

    def __repr__(self):
        return f"<Report(id={self.id}, ward={self.ward_name}, severity={self.severity})>"
