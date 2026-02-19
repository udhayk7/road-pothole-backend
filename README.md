# Predictive Road Infrastructure System - Backend

A FastAPI-based backend for managing road infrastructure reports (potholes, cracks, etc.) with intelligent clustering and priority-based repair scheduling.

## Features

- **Report Management**: Submit and track road infrastructure issues
- **DBSCAN Clustering**: Automatically groups nearby reports into clusters
- **Priority Scoring**: Calculates repair priority based on severity and report density
- **Status Tracking**: Track cluster repair status (pending, in_progress, completed)
- **AI-Powered Summaries**: Generate risk assessments using Google Gemini AI
- **Dashboard Metrics**: Aggregated KPIs including risk index for frontend dashboards

## Project Structure

```
backend/
├── main.py              # FastAPI application and endpoints
├── database.py          # SQLAlchemy database configuration
├── models.py            # ORM models (Report, Cluster)
├── schemas.py           # Pydantic validation schemas
├── services/
│   ├── clustering.py    # DBSCAN clustering logic
│   ├── priority.py      # Priority calculation service
│   └── gemini.py        # Google Gemini AI integration
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variables template
└── README.md
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

4. Run the server:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Reports

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/report` | Create a new report |
| GET | `/reports` | Get all reports |

### Clusters

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/clusters` | Get all clusters with reports |
| GET | `/priority` | Get clusters sorted by priority |
| GET | `/cluster/{id}` | Get specific cluster |
| PATCH | `/cluster/{id}/status` | Update cluster status |
| POST | `/cluster/{id}/summary` | Generate AI risk assessment |

### Dashboard

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/dashboard/metrics` | Get aggregated dashboard metrics |

### Utility

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/recluster` | Manually trigger reclustering |

## API Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Example Usage

### Create a Report

```bash
curl -X POST "http://localhost:8000/report" \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 12.9716,
    "longitude": 77.5946,
    "severity": "high",
    "image_path": "/uploads/pothole_001.jpg"
  }'
```

### Get Priority-Sorted Clusters

```bash
curl "http://localhost:8000/priority"
```

### Update Cluster Status

```bash
curl -X PATCH "http://localhost:8000/cluster/1/status" \
  -H "Content-Type: application/json" \
  -d '{"status": "in_progress"}'
```

### Generate AI Summary

```bash
curl -X POST "http://localhost:8000/cluster/1/summary"
```

## Clustering Algorithm

The system uses DBSCAN (Density-Based Spatial Clustering of Applications with Noise) to group nearby reports:

- **eps**: 0.0001 (approximately 11 meters at the equator)
- **min_samples**: 2 (minimum reports to form a cluster)

## Priority Calculation

Priority score is calculated as:

```
priority_score = avg_severity × report_count
```

Where severity is mapped as:
- low = 1
- medium = 2
- high = 3

Higher priority scores indicate areas that should be addressed first.

## Database

The system uses SQLite for data persistence. The database file (`road_infrastructure.db`) is created automatically on first run.

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API key for AI summaries | Yes (for `/cluster/{id}/summary`) |

Get your Gemini API key from: https://makersuite.google.com/app/apikey

## License

MIT
