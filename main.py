"""
Main FastAPI application for the Predictive Road Infrastructure System.
Provides REST API endpoints for managing road infrastructure reports and clusters.
"""

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import os
import uuid
from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from typing import List, Optional

from database import engine, get_db, Base
from models import Report, Cluster
from schemas import (
    ReportCreate,
    ReportResponse,
    ClusterResponse,
    ClusterStatusUpdate,
    ClusterWithReports,
    MessageResponse,
    ClusterSummaryResponse,
    DashboardMetrics,
    WardDistribution,
    ImageAnalysisResponse,
    WardInfo,
    WardListResponse,
    ContractorName,
)
from services.clustering import run_clustering, update_cluster_weather, update_all_clusters_weather
from services.priority import get_clusters_by_priority
from services.gemini import generate_cluster_summary
from services.osm_lookup import get_road_info
from services.weather import get_rain_factor, get_weather_summary
from services.ward_lookup import get_ward_name, get_ward_details, get_all_wards, get_wards_by_district
from services.lightweight_inference import analyze_image as lightweight_analyze_image

# Create uploads directory if it doesn't exist
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize FastAPI application
app = FastAPI(
    title="Predictive Road Infrastructure System",
    description="API for managing road infrastructure reports and prioritizing repairs using clustering algorithms",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving uploaded images
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


# ============== Health Check ==============

@app.get("/", tags=["Health"])
def health_check():
    """Health check endpoint to verify API is running."""
    return {"status": "healthy", "service": "Predictive Road Infrastructure System"}


# ============== Report Endpoints ==============

@app.post(
    "/report",
    response_model=ReportResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Reports"]
)
async def create_report(report: ReportCreate, db: Session = Depends(get_db)):
    """
    Create a new road infrastructure report.
    
    After saving the report, automatically triggers clustering to update
    cluster assignments and priority scores. Also fetches road and ward information.
    
    Args:
        report: Report data including location, severity, and optional image path
        db: Database session (injected)
        
    Returns:
        Created report with assigned ID, cluster, road, and ward information
    """
    # Fetch road information from OpenStreetMap
    road_info = await get_road_info(report.latitude, report.longitude)
    
    # Get ward name for the location
    ward_name = get_ward_name(report.latitude, report.longitude)
    
    # Create new report instance
    db_report = Report(
        latitude=report.latitude,
        longitude=report.longitude,
        severity=report.severity.value,
        image_path=report.image_path,
        road_name=road_info["road_name"],
        road_type=road_info["road_type"],
        ward_name=ward_name
    )
    
    db.add(db_report)
    db.commit()
    db.refresh(db_report)
    
    # Run clustering to update cluster assignments
    run_clustering(db)
    
    # Refresh to get updated cluster_id after clustering
    db.refresh(db_report)
    
    return db_report


@app.get(
    "/reports",
    response_model=List[ReportResponse],
    tags=["Reports"]
)
def get_all_reports(db: Session = Depends(get_db)):
    """
    Retrieve all road infrastructure reports.
    
    Args:
        db: Database session (injected)
        
    Returns:
        List of all reports
    """
    return db.query(Report).order_by(Report.created_at.desc()).all()


# Demo seed data (Kerala: Kochi, Trivandrum, Kozhikode, Thrissur, Kollam)
DEMO_REPORTS = [
    {"latitude": 9.9312, "longitude": 76.2673, "severity": "high", "road_name": "MG Road Kochi", "road_type": "primary"},
    {"latitude": 8.5241, "longitude": 76.9366, "severity": "medium", "road_name": "NH 66 Trivandrum Bypass", "road_type": "primary"},
    {"latitude": 11.2588, "longitude": 75.7804, "severity": "low", "road_name": "SM Street Kozhikode", "road_type": "secondary"},
    {"latitude": 10.5276, "longitude": 76.2144, "severity": "high", "road_name": "Swaraj Round Thrissur", "road_type": "tertiary"},
    {"latitude": 8.8932, "longitude": 76.6141, "severity": "medium", "road_name": "Kollam Beach Road", "road_type": "secondary"},
    # Piravom (Chinmaya area)
    {"latitude": 9.8666, "longitude": 76.4922, "road_name": "Chinmaya Road, Piravom", "severity": "high", "road_type": "secondary", "confidence_score": 0.92},
    {"latitude": 9.8672, "longitude": 76.4915, "road_name": "Piravom Town Junction", "severity": "medium", "road_type": "secondary", "confidence_score": 0.68},
]


@app.post(
    "/seed-demo-data",
    tags=["Reports"],
)
def seed_demo_data(db: Session = Depends(get_db)):
    """
    Temporary endpoint: insert demo reports in Kerala + Piravom (Chinmaya area).
    After inserting, runs run_clustering(db). Returns status.
    """
    for r in DEMO_REPORTS:
        db.add(Report(
            latitude=r["latitude"],
            longitude=r["longitude"],
            severity=r["severity"],
            image_path=None,
            road_name=r["road_name"],
            road_type=r.get("road_type", "unknown"),
            ward_name="Unknown",
            confidence_score=r.get("confidence_score"),
        ))
    db.commit()
    run_clustering(db)
    return {"status": "Demo Kerala + Piravom data created"}


@app.post(
    "/reports/upload",
    response_model=ReportResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Reports"]
)
async def create_report_with_image(
    image: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    severity: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    Create a new road infrastructure report with an uploaded image.
    
    Args:
        image: Image file of the road damage
        latitude: GPS latitude coordinate
        longitude: GPS longitude coordinate
        severity: Severity level (low, medium, high)
        db: Database session (injected)
        
    Returns:
        Created report with assigned ID, cluster, road, and ward information
    """
    # Generate unique filename
    file_extension = os.path.splitext(image.filename)[1] if image.filename else ".jpg"
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    # Save uploaded file
    with open(file_path, "wb") as buffer:
        content = await image.read()
        buffer.write(content)
    
    # Fetch road information from OpenStreetMap
    road_info = await get_road_info(latitude, longitude)
    
    # Get ward name for the location
    ward_name = get_ward_name(latitude, longitude)
    
    # Create new report instance
    db_report = Report(
        latitude=latitude,
        longitude=longitude,
        severity=severity,
        image_path=f"/uploads/{unique_filename}",
        road_name=road_info["road_name"],
        road_type=road_info["road_type"],
        ward_name=ward_name
    )
    
    db.add(db_report)
    db.commit()
    db.refresh(db_report)
    
    # Run clustering to update cluster assignments
    run_clustering(db)
    
    # Refresh to get updated cluster_id after clustering
    db.refresh(db_report)
    
    return db_report


@app.post(
    "/analyze",
    response_model=ImageAnalysisResponse,
    tags=["Reports"]
)
async def analyze_image(image: UploadFile = File(...)):
    """
    Analyze an uploaded image for road damage using lightweight edge-based inference.
    
    Saves the image locally, runs OpenCV edge analysis, and returns
    severity (low/medium/high) and confidence_score.
    
    Args:
        image: Image file to analyze
        
    Returns:
        Detected severity and confidence score
    """
    if not image.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image file is required"
        )
    
    # Save uploaded image to a temporary path
    ext = os.path.splitext(image.filename)[1] or ".jpg"
    safe_name = f"{uuid.uuid4().hex}{ext}"
    saved_path = os.path.join(UPLOAD_DIR, safe_name)
    
    try:
        contents = await image.read()
        with open(saved_path, "wb") as f:
            f.write(contents)
    except OSError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded image: {e}"
        ) from e
    
    try:
        result = lightweight_analyze_image(saved_path)
        return ImageAnalysisResponse(
            severity=result["severity"],
            confidence_score=result["confidence_score"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) from e
    finally:
        # Clean up saved file
        try:
            if os.path.exists(saved_path):
                os.remove(saved_path)
        except OSError:
            pass


# ============== Cluster Endpoints ==============

@app.get(
    "/clusters",
    response_model=List[ClusterWithReports],
    tags=["Clusters"]
)
def get_all_clusters(db: Session = Depends(get_db)):
    """
    Retrieve all clusters with their associated reports.
    
    Args:
        db: Database session (injected)
        
    Returns:
        List of all clusters including report details
    """
    return db.query(Cluster).all()


@app.get(
    "/priority",
    response_model=List[ClusterResponse],
    tags=["Clusters"]
)
def get_clusters_by_priority_endpoint(db: Session = Depends(get_db)):
    """
    Retrieve all clusters ordered by priority score (highest first).
    
    Priority score is calculated as: avg_severity * report_count
    Higher scores indicate areas that should be addressed first.
    
    Args:
        db: Database session (injected)
        
    Returns:
        List of clusters sorted by priority_score descending
    """
    return get_clusters_by_priority(db)


@app.get(
    "/contractors",
    response_model=List[ContractorName],
    tags=["Clusters"]
)
def get_contractors(db: Session = Depends(get_db)):
    """
    Get all unique contractor names from the Cluster table.
    Returns empty list if no clusters exist.
    """
    rows = db.query(Cluster.contractor_name).distinct().all()
    names = [r[0] for r in rows if r[0]]
    return [ContractorName(name=n) for n in sorted(names)]


@app.patch(
    "/cluster/{cluster_id}/status",
    response_model=ClusterResponse,
    tags=["Clusters"]
)
def update_cluster_status(
    cluster_id: int,
    status_update: ClusterStatusUpdate,
    db: Session = Depends(get_db)
):
    """
    Update the status of a specific cluster.
    
    Valid statuses: pending, in_progress, completed
    
    Args:
        cluster_id: ID of the cluster to update
        status_update: New status value
        db: Database session (injected)
        
    Returns:
        Updated cluster information
        
    Raises:
        HTTPException: If cluster is not found
    """
    cluster = db.query(Cluster).filter(Cluster.id == cluster_id).first()
    
    if not cluster:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cluster with id {cluster_id} not found"
        )
    
    cluster.status = status_update.status.value
    db.commit()
    db.refresh(cluster)
    
    return cluster


@app.get(
    "/cluster/{cluster_id}",
    response_model=ClusterWithReports,
    tags=["Clusters"]
)
def get_cluster_by_id(cluster_id: int, db: Session = Depends(get_db)):
    """
    Retrieve a specific cluster by ID with its associated reports.
    
    Args:
        cluster_id: ID of the cluster to retrieve
        db: Database session (injected)
        
    Returns:
        Cluster information with associated reports
        
    Raises:
        HTTPException: If cluster is not found
    """
    cluster = db.query(Cluster).filter(Cluster.id == cluster_id).first()
    
    if not cluster:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cluster with id {cluster_id} not found"
        )
    
    return cluster


def _generate_rule_based_summary(cluster: Cluster) -> str:
    """Generate summary from cluster data using rule-based logic (no external APIs)."""
    parts = []
    if cluster.avg_severity >= 2.5:
        parts.append("High severity damage detected.")
    elif cluster.avg_severity >= 1.5:
        parts.append("Moderate severity damage detected.")
    else:
        parts.append("Low severity damage detected.")
    if cluster.priority_score >= 4:
        parts.append("Urgent intervention required.")
    if cluster.report_count > 3:
        parts.append("Multiple citizen complaints detected.")
    road_type = cluster.dominant_road_type or "unknown"
    ward = cluster.ward_name or "Unknown"
    parts.append(f"Location: {ward}. Road type: {road_type}. {cluster.report_count} report(s), priority score {cluster.priority_score}.")
    return " ".join(parts)


@app.post(
    "/cluster/{cluster_id}/summary",
    response_model=ClusterSummaryResponse,
    tags=["Clusters"]
)
def get_cluster_summary(cluster_id: int, db: Session = Depends(get_db)):
    """
    Generate a rule-based risk summary for a cluster (no external APIs).
    Uses avg_severity, report_count, priority_score, dominant_road_type, ward_name.
    """
    cluster = db.query(Cluster).filter(Cluster.id == cluster_id).first()
    if not cluster:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cluster with id {cluster_id} not found"
        )
    summary = _generate_rule_based_summary(cluster)
    return ClusterSummaryResponse(
        cluster_id=cluster.id,
        summary=summary,
        avg_severity=cluster.avg_severity,
        report_count=cluster.report_count,
        priority_score=cluster.priority_score
    )


# ============== Ward Endpoints ==============

@app.get(
    "/wards",
    response_model=WardListResponse,
    tags=["Wards"]
)
def list_all_wards():
    """
    Get list of all available wards in Kerala.
    
    Returns:
        List of ward information
    """
    wards = get_all_wards()
    ward_list = [
        WardInfo(
            ward_name=w.get("ward_name", ""),
            ward_id=w.get("ward_id", ""),
            district=w.get("district", ""),
            city=w.get("city", "")
        )
        for w in wards
    ]
    return WardListResponse(wards=ward_list, total=len(ward_list))


@app.get(
    "/wards/district/{district}",
    response_model=WardListResponse,
    tags=["Wards"]
)
def list_wards_by_district(district: str):
    """
    Get all wards in a specific district.
    
    Args:
        district: District name (e.g., Ernakulam, Thiruvananthapuram)
        
    Returns:
        List of wards in the district
    """
    wards = get_wards_by_district(district)
    ward_list = [
        WardInfo(
            ward_name=w.get("ward_name", ""),
            ward_id=w.get("ward_id", ""),
            district=w.get("district", ""),
            city=w.get("city", "")
        )
        for w in wards
    ]
    return WardListResponse(wards=ward_list, total=len(ward_list))


@app.get(
    "/ward/lookup/{latitude}/{longitude}",
    response_model=WardInfo,
    tags=["Wards"]
)
def lookup_ward(latitude: float, longitude: float):
    """
    Look up the ward for a specific location.
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        
    Returns:
        Ward information for the location
    """
    details = get_ward_details(latitude, longitude)
    return WardInfo(
        ward_name=details["ward_name"],
        ward_id=details["ward_id"],
        district=details["district"],
        city=details["city"],
        found=details["found"]
    )


@app.get(
    "/clusters/ward/{ward_name}",
    response_model=List[ClusterWithReports],
    tags=["Clusters"]
)
def get_clusters_by_ward(ward_name: str, db: Session = Depends(get_db)):
    """
    Get all clusters in a specific ward.
    
    Args:
        ward_name: Name of the ward to filter by
        db: Database session (injected)
        
    Returns:
        List of clusters in the specified ward
    """
    return db.query(Cluster).filter(Cluster.ward_name == ward_name).all()


@app.get(
    "/reports/ward/{ward_name}",
    response_model=List[ReportResponse],
    tags=["Reports"]
)
def get_reports_by_ward(ward_name: str, db: Session = Depends(get_db)):
    """
    Get all reports in a specific ward.
    
    Args:
        ward_name: Name of the ward to filter by
        db: Database session (injected)
        
    Returns:
        List of reports in the specified ward
    """
    return db.query(Report).filter(Report.ward_name == ward_name).order_by(Report.created_at.desc()).all()


# ============== Dashboard Endpoints ==============

@app.get(
    "/dashboard/metrics",
    response_model=DashboardMetrics,
    tags=["Dashboard"]
)
def get_dashboard_metrics(ward_name: Optional[str] = None, db: Session = Depends(get_db)):
    """
    Get aggregated metrics for the dashboard.
    
    Returns key performance indicators including:
    - Total clusters and reports
    - Status breakdown (pending, in_progress, completed)
    - High priority cluster count
    - Average priority score
    - Ward distribution
    - Overall risk index (0-100 scale)
    
    Supports optional ward filtering for ward-specific metrics.
    
    Args:
        ward_name: Optional ward name to filter metrics by
        db: Database session (injected)
        
    Returns:
        Dashboard metrics for frontend display
    """
    # Base queries - apply ward filter if provided
    cluster_query = db.query(Cluster)
    report_query = db.query(Report)
    
    if ward_name:
        cluster_query = cluster_query.filter(Cluster.ward_name == ward_name)
        report_query = report_query.filter(Report.ward_name == ward_name)
    
    # Get all clusters for calculations
    clusters = cluster_query.all()
    total_clusters = len(clusters)
    
    # Status counts
    pending = sum(1 for c in clusters if c.status == "pending")
    in_progress = sum(1 for c in clusters if c.status == "in_progress")
    completed = sum(1 for c in clusters if c.status == "completed")
    
    # High priority: clusters with priority_score >= 6
    high_priority_clusters = sum(1 for c in clusters if c.priority_score >= 6)
    
    # Total reports
    total_reports = report_query.count()
    
    # Calculate average priority
    average_priority = 0.0
    if total_clusters > 0:
        average_priority = round(sum(c.priority_score for c in clusters) / total_clusters, 2)
    
    # Calculate ward distribution
    ward_stats = {}
    for cluster in clusters:
        ward = cluster.ward_name or "Unknown"
        if ward not in ward_stats:
            ward_stats[ward] = {
                "cluster_count": 0,
                "report_count": 0,
                "total_priority": 0.0
            }
        ward_stats[ward]["cluster_count"] += 1
        ward_stats[ward]["report_count"] += cluster.report_count
        ward_stats[ward]["total_priority"] += cluster.priority_score
    
    ward_distribution = [
        WardDistribution(
            ward_name=ward,
            cluster_count=stats["cluster_count"],
            report_count=stats["report_count"],
            avg_priority=round(stats["total_priority"] / stats["cluster_count"], 2) if stats["cluster_count"] > 0 else 0.0
        )
        for ward, stats in sorted(ward_stats.items(), key=lambda x: x[1]["cluster_count"], reverse=True)
    ]
    
    # Calculate risk index (0-100 scale)
    risk_index = 0.0
    if total_clusters > 0:
        # Factors for risk calculation:
        # 1. Average priority score (normalized to 0-40)
        priority_factor = min(average_priority * 4, 40)  # Cap at 40 points
        
        # 2. Pending ratio (0-30)
        pending_ratio = pending / total_clusters
        pending_factor = pending_ratio * 30
        
        # 3. High priority ratio (0-30)
        high_priority_ratio = high_priority_clusters / total_clusters
        high_priority_factor = high_priority_ratio * 30
        
        risk_index = round(priority_factor + pending_factor + high_priority_factor, 1)
    
    return DashboardMetrics(
        total_clusters=total_clusters,
        high_priority_clusters=high_priority_clusters,
        in_progress=in_progress,
        completed=completed,
        pending=pending,
        total_reports=total_reports,
        average_priority=average_priority,
        risk_index=risk_index,
        ward_distribution=ward_distribution
    )


# ============== Utility Endpoints ==============

@app.post(
    "/recluster",
    response_model=MessageResponse,
    tags=["Utility"]
)
def trigger_reclustering(db: Session = Depends(get_db)):
    """
    Manually trigger reclustering of all reports.
    
    Useful when DBSCAN parameters are adjusted or for maintenance.
    
    Args:
        db: Database session (injected)
        
    Returns:
        Success message with cluster count
    """
    run_clustering(db)
    cluster_count = db.query(Cluster).count()
    
    return MessageResponse(
        message="Clustering completed successfully",
        data={"cluster_count": cluster_count}
    )


@app.post(
    "/weather/refresh",
    response_model=MessageResponse,
    tags=["Utility"]
)
async def refresh_weather_data(db: Session = Depends(get_db)):
    """
    Refresh weather data and recalculate priorities for all clusters.
    
    Fetches current rainfall data from OpenWeatherMap and updates
    rain_factor and priority_score for each cluster.
    
    Args:
        db: Database session (injected)
        
    Returns:
        Success message with updated cluster count
    """
    await update_all_clusters_weather(db)
    cluster_count = db.query(Cluster).count()
    
    return MessageResponse(
        message="Weather data refreshed and priorities recalculated",
        data={"clusters_updated": cluster_count}
    )


@app.get(
    "/weather/{latitude}/{longitude}",
    tags=["Utility"]
)
async def get_weather_info(latitude: float, longitude: float):
    """
    Get current weather information for a location.
    
    Useful for checking rainfall conditions in Kerala.
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        
    Returns:
        Weather summary including rainfall and rain factor
    """
    weather = await get_weather_summary(latitude, longitude)
    return weather


@app.post(
    "/cluster/{cluster_id}/refresh-weather",
    response_model=ClusterResponse,
    tags=["Clusters"]
)
async def refresh_cluster_weather(cluster_id: int, db: Session = Depends(get_db)):
    """
    Refresh weather data for a specific cluster.
    
    Updates the rain_factor and recalculates priority_score based on
    current rainfall at the cluster's location.
    
    Args:
        cluster_id: ID of the cluster to update
        db: Database session (injected)
        
    Returns:
        Updated cluster information
    """
    cluster = db.query(Cluster).filter(Cluster.id == cluster_id).first()
    
    if not cluster:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cluster with id {cluster_id} not found"
        )
    
    await update_cluster_weather(db, cluster_id)
    db.refresh(cluster)
    
    return cluster


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
