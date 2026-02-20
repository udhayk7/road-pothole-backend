"""
Pydantic schemas for request/response validation and serialization.
Provides data validation and API documentation for the FastAPI endpoints.
"""

from datetime import datetime
from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field


class SeverityLevel(str, Enum):
    """Enumeration of valid severity levels for reports."""
    low = "low"
    medium = "medium"
    high = "high"


class ClusterStatus(str, Enum):
    """Enumeration of valid cluster statuses."""
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"


# ============== Report Schemas ==============

class ReportCreate(BaseModel):
    """Schema for creating a new report."""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")
    severity: SeverityLevel = Field(..., description="Severity level of the issue")
    image_path: Optional[str] = Field(None, description="Path to the uploaded image")

    class Config:
        json_schema_extra = {
            "example": {
                "latitude": 12.9716,
                "longitude": 77.5946,
                "severity": "medium",
                "image_path": "/uploads/pothole_001.jpg"
            }
        }


class ReportResponse(BaseModel):
    """Schema for report response."""
    id: int
    latitude: float
    longitude: float
    severity: str
    image_path: Optional[str]
    road_name: Optional[str] = "Unknown"
    road_type: Optional[str] = "unknown"
    ward_name: Optional[str] = "Unknown"
    created_at: datetime
    cluster_id: Optional[int]

    class Config:
        from_attributes = True


# ============== Cluster Schemas ==============

class ClusterResponse(BaseModel):
    """Schema for cluster response with road type, weather, ward, and contractor for Kerala."""
    id: int
    avg_severity: float
    report_count: int
    priority_score: float
    dominant_road_type: str = "unknown"
    road_type_weight: float = 1.0
    rain_factor: float = 1.0
    ward_name: str = "Unknown"
    center_lat: Optional[float] = None
    center_lon: Optional[float] = None
    status: str
    contractor_name: Optional[str] = None

    class Config:
        from_attributes = True


class ClusterWithReports(ClusterResponse):
    """Schema for cluster response including associated reports."""
    reports: List[ReportResponse] = []

    class Config:
        from_attributes = True


class ClusterStatusUpdate(BaseModel):
    """Schema for updating cluster status."""
    status: ClusterStatus = Field(..., description="New status for the cluster")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "in_progress"
            }
        }


# ============== Response Wrappers ==============

class MessageResponse(BaseModel):
    """Generic message response schema."""
    message: str
    data: Optional[dict] = None


class ClusterSummaryResponse(BaseModel):
    """Schema for AI-generated cluster summary response."""
    cluster_id: int
    summary: str
    avg_severity: float
    report_count: int
    priority_score: float


# ============== Image Analysis Schemas ==============

class ImageAnalysisResponse(BaseModel):
    """Schema for image analysis response."""
    severity: str = Field(..., description="Detected severity level")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence score of the detection")

    class Config:
        json_schema_extra = {
            "example": {
                "severity": "high",
                "confidence_score": 0.92
            }
        }


# ============== Ward Schemas ==============

class WardInfo(BaseModel):
    """Schema for ward information."""
    ward_name: str
    ward_id: str
    district: str
    city: str
    found: bool = True


class WardListResponse(BaseModel):
    """Schema for list of wards."""
    wards: List[WardInfo]
    total: int


# ============== Dashboard Schemas ==============

class WardDistribution(BaseModel):
    """Schema for ward distribution in dashboard."""
    ward_name: str
    cluster_count: int
    report_count: int
    avg_priority: float


class DashboardMetrics(BaseModel):
    """Schema for dashboard metrics response."""
    total_clusters: int = Field(..., description="Total number of clusters")
    high_priority_clusters: int = Field(..., description="Clusters with priority score >= 6")
    in_progress: int = Field(..., description="Clusters currently being worked on")
    completed: int = Field(..., description="Clusters that have been resolved")
    pending: int = Field(..., description="Clusters awaiting action")
    total_reports: int = Field(..., description="Total number of reports submitted")
    average_priority: float = Field(..., description="Average priority score across all clusters")
    risk_index: float = Field(..., description="Overall risk index (0-100 scale)")
    ward_distribution: List[WardDistribution] = Field(default=[], description="Distribution of clusters by ward")

    class Config:
        json_schema_extra = {
            "example": {
                "total_clusters": 15,
                "high_priority_clusters": 5,
                "in_progress": 3,
                "completed": 2,
                "pending": 10,
                "total_reports": 47,
                "average_priority": 7.5,
                "risk_index": 68.5,
                "ward_distribution": [
                    {"ward_name": "Ernakulam South", "cluster_count": 3, "report_count": 8, "avg_priority": 8.2},
                    {"ward_name": "Fort Kochi", "cluster_count": 2, "report_count": 5, "avg_priority": 6.1}
                ]
            }
        }
