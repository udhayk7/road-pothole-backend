"""
Google Gemini AI service for generating cluster risk assessments and repair recommendations.
"""

import os
import google.generativeai as genai
from typing import Dict, Any, Optional


# Configure Gemini API with environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def get_gemini_model():
    """
    Get configured Gemini model instance.
    
    Returns:
        GenerativeModel instance or None if API key not configured
    """
    if not GEMINI_API_KEY:
        return None
    return genai.GenerativeModel("gemini-1.5-flash")


def format_cluster_data(cluster_data: Dict[str, Any]) -> str:
    """
    Format cluster data into a readable string for the AI prompt.
    
    Args:
        cluster_data: Dictionary containing cluster information
        
    Returns:
        Formatted string representation of cluster data
    """
    reports_info = []
    for report in cluster_data.get("reports", []):
        reports_info.append(
            f"  - Report #{report['id']}: severity={report['severity']}, "
            f"location=({report['latitude']:.6f}, {report['longitude']:.6f})"
        )
    
    reports_str = "\n".join(reports_info) if reports_info else "  No reports"
    
    return f"""
Cluster ID: {cluster_data.get('id')}
Average Severity: {cluster_data.get('avg_severity')} (scale: 1=low, 2=medium, 3=high)
Number of Reports: {cluster_data.get('report_count')}
Priority Score: {cluster_data.get('priority_score')}
Current Status: {cluster_data.get('status')}

Individual Reports:
{reports_str}
"""


async def generate_cluster_summary(cluster_data: Dict[str, Any]) -> Optional[str]:
    """
    Generate a risk explanation and repair recommendation for a cluster using Gemini AI.
    
    Args:
        cluster_data: Dictionary containing cluster information including:
            - id: Cluster ID
            - avg_severity: Average severity score
            - report_count: Number of reports in cluster
            - priority_score: Calculated priority score
            - status: Current cluster status
            - reports: List of report details
            
    Returns:
        Generated summary text or None if generation fails
        
    Raises:
        ValueError: If Gemini API key is not configured
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    
    model = get_gemini_model()
    if not model:
        raise ValueError("Failed to initialize Gemini model")
    
    formatted_data = format_cluster_data(cluster_data)
    
    prompt = f"""You are a road infrastructure expert analyzing road damage data.

Generate a risk explanation and recommended repair action for this road damage cluster.

{formatted_data}

Please provide:
1. **Risk Assessment**: Explain the severity and potential dangers of this cluster
2. **Impact Analysis**: Describe potential impact on traffic, vehicles, and pedestrians
3. **Recommended Actions**: Specific repair actions with priority level
4. **Resource Estimate**: Approximate resources needed (crew size, equipment, time)

Keep the response concise but informative, suitable for a maintenance team report."""

    try:
        response = await model.generate_content_async(prompt)
        return response.text
    except Exception as e:
        raise RuntimeError(f"Failed to generate summary: {str(e)}")
