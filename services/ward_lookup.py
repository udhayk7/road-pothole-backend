"""
Ward lookup service for tagging reports with Kerala ward information.
Uses shapely to perform point-in-polygon checks against ward boundaries.
"""

import json
import os
from typing import Dict, Optional, List
from shapely.geometry import Point, shape
from functools import lru_cache

# Path to ward GeoJSON file
WARDS_GEOJSON_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 
    "data", 
    "wards.geojson"
)


@lru_cache(maxsize=1)
def load_ward_data() -> List[Dict]:
    """
    Load and cache ward polygon data from GeoJSON file.
    
    Returns:
        List of ward features with geometry and properties
    """
    try:
        with open(WARDS_GEOJSON_PATH, 'r') as f:
            geojson_data = json.load(f)
            
        wards = []
        for feature in geojson_data.get("features", []):
            ward_info = {
                "properties": feature.get("properties", {}),
                "geometry": shape(feature.get("geometry", {}))
            }
            wards.append(ward_info)
        
        return wards
    except FileNotFoundError:
        print(f"Warning: Ward GeoJSON file not found at {WARDS_GEOJSON_PATH}")
        return []
    except Exception as e:
        print(f"Warning: Error loading ward data: {e}")
        return []


def get_ward_for_location(latitude: float, longitude: float) -> Optional[Dict]:
    """
    Find the ward that contains the given coordinates.
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        
    Returns:
        Ward properties dict if found, None otherwise
    """
    wards = load_ward_data()
    point = Point(longitude, latitude)  # Note: shapely uses (lon, lat) order
    
    for ward in wards:
        if ward["geometry"].contains(point):
            return ward["properties"]
    
    return None


def get_ward_name(latitude: float, longitude: float) -> str:
    """
    Get the ward name for a given location.
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        
    Returns:
        Ward name string, or "Unknown" if not in any ward
    """
    ward_info = get_ward_for_location(latitude, longitude)
    
    if ward_info:
        return ward_info.get("ward_name", "Unknown")
    
    return "Unknown"


def get_ward_details(latitude: float, longitude: float) -> Dict:
    """
    Get detailed ward information for a location.
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        
    Returns:
        Dictionary with ward details
    """
    ward_info = get_ward_for_location(latitude, longitude)
    
    if ward_info:
        return {
            "ward_name": ward_info.get("ward_name", "Unknown"),
            "ward_id": ward_info.get("ward_id", ""),
            "district": ward_info.get("district", ""),
            "city": ward_info.get("city", ""),
            "found": True
        }
    
    return {
        "ward_name": "Unknown",
        "ward_id": "",
        "district": "",
        "city": "",
        "found": False
    }


def get_all_wards() -> List[Dict]:
    """
    Get list of all available wards.
    
    Returns:
        List of ward property dictionaries
    """
    wards = load_ward_data()
    return [ward["properties"] for ward in wards]


def get_wards_by_district(district: str) -> List[Dict]:
    """
    Get all wards in a specific district.
    
    Args:
        district: District name
        
    Returns:
        List of ward properties in the district
    """
    wards = load_ward_data()
    return [
        ward["properties"] 
        for ward in wards 
        if ward["properties"].get("district", "").lower() == district.lower()
    ]


def get_wards_by_city(city: str) -> List[Dict]:
    """
    Get all wards in a specific city.
    
    Args:
        city: City name
        
    Returns:
        List of ward properties in the city
    """
    wards = load_ward_data()
    return [
        ward["properties"] 
        for ward in wards 
        if ward["properties"].get("city", "").lower() == city.lower()
    ]
