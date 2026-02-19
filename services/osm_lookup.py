"""
OpenStreetMap lookup service for fetching road information.
Uses the Overpass API to find nearest road data based on coordinates.
"""

import httpx
from typing import Dict, Optional

# Overpass API endpoint
OVERPASS_API_URL = "https://overpass-api.de/api/interpreter"

# Search radius in meters for finding nearby roads
SEARCH_RADIUS = 50


async def get_road_info(latitude: float, longitude: float) -> Dict[str, str]:
    """
    Fetch road information from OpenStreetMap for given coordinates.
    
    Uses Overpass API to query for the nearest highway/road within
    a specified radius of the given location.
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        
    Returns:
        Dictionary containing:
        - road_name: Name of the road (or "Unnamed Road" if not available)
        - road_type: Type of highway (e.g., residential, primary, secondary)
    """
    # Overpass QL query to find nearest roads
    query = f"""
    [out:json][timeout:10];
    (
      way["highway"](around:{SEARCH_RADIUS},{latitude},{longitude});
    );
    out body;
    >;
    out skel qt;
    """
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                OVERPASS_API_URL,
                data={"data": query}
            )
            response.raise_for_status()
            data = response.json()
            
            return _parse_road_data(data)
            
    except httpx.TimeoutException:
        return {"road_name": "Unknown", "road_type": "unknown"}
    except httpx.HTTPStatusError:
        return {"road_name": "Unknown", "road_type": "unknown"}
    except Exception:
        return {"road_name": "Unknown", "road_type": "unknown"}


def _parse_road_data(data: Dict) -> Dict[str, str]:
    """
    Parse Overpass API response to extract road information.
    
    Args:
        data: JSON response from Overpass API
        
    Returns:
        Dictionary with road_name and road_type
    """
    elements = data.get("elements", [])
    
    # Find the first way element with highway tag
    for element in elements:
        if element.get("type") == "way" and "tags" in element:
            tags = element["tags"]
            
            if "highway" in tags:
                road_name = tags.get("name", "Unnamed Road")
                road_type = tags.get("highway", "unknown")
                
                return {
                    "road_name": road_name,
                    "road_type": road_type
                }
    
    return {"road_name": "Unknown", "road_type": "unknown"}


async def get_road_info_batch(coordinates: list) -> list:
    """
    Fetch road information for multiple coordinates.
    
    Args:
        coordinates: List of (latitude, longitude) tuples
        
    Returns:
        List of road info dictionaries
    """
    results = []
    for lat, lon in coordinates:
        info = await get_road_info(lat, lon)
        results.append(info)
    return results
