"""
Weather service for fetching rainfall data from Open-Meteo API.
Used to adjust priority scores based on recent rainfall in Kerala.
Heavy rain accelerates road damage, so affected areas get higher priority.

Open-Meteo is a free, no-API-key-required weather service.
"""

import httpx
from typing import Dict, Optional

# Open-Meteo API configuration
OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"

# Rain factor multipliers based on hourly rainfall
HEAVY_RAIN_THRESHOLD_MM = 2.0
LIGHT_RAIN_THRESHOLD_MM = 0.0

HEAVY_RAIN_FACTOR = 1.5
LIGHT_RAIN_FACTOR = 1.2
NO_RAIN_FACTOR = 1.0


async def get_hourly_rain(latitude: float, longitude: float) -> Optional[float]:
    """
    Fetch latest hourly rain value from Open-Meteo API.
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        
    Returns:
        Latest hourly rain value in mm, or None if request fails
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                OPEN_METEO_BASE_URL,
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "hourly": "rain",
                    "forecast_days": 1
                }
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract hourly rain data
            hourly = data.get("hourly", {})
            rain_values = hourly.get("rain", [])
            
            if rain_values:
                # Get the latest non-null rain value
                for rain in reversed(rain_values):
                    if rain is not None:
                        return float(rain)
            
            return 0.0
            
    except Exception:
        return None


async def get_rain_factor(latitude: float, longitude: float) -> float:
    """
    Calculate rain factor for priority scoring based on current rainfall.
    
    Logic:
    - rain > 2mm: rain_factor = 1.5 (heavy rain)
    - rain > 0mm: rain_factor = 1.2 (light rain)
    - rain = 0mm: rain_factor = 1.0 (no rain)
    
    Heavy rainfall accelerates road damage (potholes expand, cracks worsen),
    so areas with recent rain get higher repair priority.
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        
    Returns:
        Rain factor multiplier (1.0, 1.2, or 1.5)
    """
    try:
        rain_mm = await get_hourly_rain(latitude, longitude)
        
        if rain_mm is None:
            return NO_RAIN_FACTOR
        
        if rain_mm > HEAVY_RAIN_THRESHOLD_MM:
            return HEAVY_RAIN_FACTOR
        elif rain_mm > LIGHT_RAIN_THRESHOLD_MM:
            return LIGHT_RAIN_FACTOR
        else:
            return NO_RAIN_FACTOR
            
    except Exception:
        return NO_RAIN_FACTOR


async def get_weather_summary(latitude: float, longitude: float) -> Dict:
    """
    Get a summary of weather conditions for a location.
    Useful for displaying in reports and dashboards.
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        
    Returns:
        Dictionary with weather summary
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                OPEN_METEO_BASE_URL,
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "hourly": "rain,temperature_2m,relative_humidity_2m",
                    "forecast_days": 1
                }
            )
            response.raise_for_status()
            data = response.json()
            
            hourly = data.get("hourly", {})
            rain_values = hourly.get("rain", [])
            temp_values = hourly.get("temperature_2m", [])
            humidity_values = hourly.get("relative_humidity_2m", [])
            
            # Get latest values
            latest_rain = 0.0
            latest_temp = None
            latest_humidity = None
            
            for rain in reversed(rain_values):
                if rain is not None:
                    latest_rain = float(rain)
                    break
            
            for temp in reversed(temp_values):
                if temp is not None:
                    latest_temp = float(temp)
                    break
            
            for humidity in reversed(humidity_values):
                if humidity is not None:
                    latest_humidity = int(humidity)
                    break
            
            # Determine rain factor
            if latest_rain > HEAVY_RAIN_THRESHOLD_MM:
                rain_factor = HEAVY_RAIN_FACTOR
                condition = "Heavy Rain"
            elif latest_rain > LIGHT_RAIN_THRESHOLD_MM:
                rain_factor = LIGHT_RAIN_FACTOR
                condition = "Light Rain"
            else:
                rain_factor = NO_RAIN_FACTOR
                condition = "Clear"
            
            return {
                "available": True,
                "rain_mm": round(latest_rain, 2),
                "rain_factor": rain_factor,
                "condition": condition,
                "temperature_c": latest_temp,
                "humidity_percent": latest_humidity
            }
            
    except Exception:
        return {
            "available": False,
            "rain_mm": 0.0,
            "rain_factor": NO_RAIN_FACTOR,
            "condition": "unknown"
        }


# Alias for backward compatibility
async def get_rainfall_factor(latitude: float, longitude: float) -> float:
    """Alias for get_rain_factor for backward compatibility."""
    return await get_rain_factor(latitude, longitude)
