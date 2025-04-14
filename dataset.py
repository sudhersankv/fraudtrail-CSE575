import numpy as np
import pandas as pd
import requests
import random
import time
import folium
from folium import Popup
import os
import math
from typing import Tuple, Dict, List, Any
from geopy.distance import geodesic

# Define 10 major US cities with their coordinates
US_CITIES = [
    {"name": "New York", "lat": 40.7128, "lon": -74.0060},
    {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437},
    {"name": "Chicago", "lat": 41.8781, "lon": -87.6298},
    {"name": "Houston", "lat": 29.7604, "lon": -95.3698},
    {"name": "Phoenix", "lat": 33.4484, "lon": -112.0740},
    {"name": "Philadelphia", "lat": 39.9526, "lon": -75.1652},
    {"name": "San Antonio", "lat": 29.4241, "lon": -98.4936},
    {"name": "San Diego", "lat": 32.7157, "lon": -117.1611},
    {"name": "Dallas", "lat": 32.7767, "lon": -96.7970},
    {"name": "Denver", "lat": 39.7392, "lon": -104.9903}
]

# Additional cities to have more variety in routes
ADDITIONAL_CITIES = [
    {"name": "Seattle", "lat": 47.6062, "lon": -122.3321},
    {"name": "Miami", "lat": 25.7617, "lon": -80.1918},
    {"name": "Atlanta", "lat": 33.7490, "lon": -84.3880},
    {"name": "Boston", "lat": 42.3601, "lon": -71.0589},
    {"name": "Las Vegas", "lat": 36.1699, "lon": -115.1398},
    {"name": "Portland", "lat": 45.5152, "lon": -122.6784},
    {"name": "Detroit", "lat": 42.3314, "lon": -83.0458},
    {"name": "San Francisco", "lat": 37.7749, "lon": -122.4194},
    {"name": "Washington DC", "lat": 38.9072, "lon": -77.0369},
    {"name": "Austin", "lat": 30.2672, "lon": -97.7431}
]

ALL_CITIES = US_CITIES + ADDITIONAL_CITIES

# Fraud patterns - each driver ID might have specific fraud behaviors
FRAUD_PATTERNS = [
    {
        "name": "route_padder",
        "description": "Takes unnecessarily longer routes to inflate mileage",
        "loop_factor": (1.5, 3.0),
        "route_similarity": (0.3, 0.5),
        "weather_impact": (0.0, 0.2),
        "traffic_impact": (0.0, 0.3),
        "driver_rating": (2.0, 3.2),
        "fulfillment_probability": 0.8  # 80% chance of fulfilling the order despite fraud
    },
    {
        "name": "time_waster",
        "description": "Reports excessive delays without justification",
        "delay_factor": (1.7, 3.0),
        "route_similarity": (0.5, 0.8),
        "weather_impact": (0.0, 0.3),
        "traffic_impact": (0.0, 0.2),
        "driver_rating": (2.5, 3.8),
        "fulfillment_probability": 0.7  # 70% chance of fulfilling the order
    },
    {
        "name": "ghost_delivery",
        "description": "Claims delivery without going to destination",
        "time_reduction": (0.2, 0.4),
        "route_similarity": (0.1, 0.3),
        "driver_rating": (1.0, 2.5),
        "fulfillment_probability": 0.1  # Only 10% chance of fulfilling the order
    }
]

def get_city_pair() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Get a random pair of different cities"""
    origin = random.choice(ALL_CITIES)
    destination = random.choice([city for city in ALL_CITIES if city != origin])
    return origin, destination

def calculate_realistic_mid_deviation(start_coords, end_coords):
    """
    Returns a realistic off-route midpoint simulating fraudulent deviation.
    The point is offset from the midpoint between start and end by a random
    distance (1 to 30 km) and direction (20 to 160 degrees).
    """
    # Calculate geographic midpoint
    mid_lat = (start_coords[0] + end_coords[0]) / 2
    mid_lon = (start_coords[1] + end_coords[1]) / 2
    
    # Randomized deviation
    deviation_km = random.uniform(1, 30)  # 1-30km deviation
    angle_degrees = random.randint(20, 160)  # Deviation angle
    
    # Convert angle to radians
    angle_radians = math.radians(angle_degrees)
    
    earth_radius = 6371.0
    
    # Approximate conversion from km to latitude/longitude degrees
    lat_change = (deviation_km / earth_radius) * (180 / math.pi)
    lon_change = (deviation_km / (earth_radius * math.cos(math.radians(mid_lat)))) * (180 / math.pi)
    
    # Calculate the deviated point for route deviation fraud
    deviated_lat = mid_lat + lat_change * math.sin(angle_radians)
    deviated_lon = mid_lon + lon_change * math.cos(angle_radians)
    
    return (deviated_lat, deviated_lon)

def create_route_deviation(origin: Tuple[float, float], destination: Tuple[float, float], 
                           deviation_type: str = "random") -> List[Tuple[float, float]]:
    """
    Create route deviations based on different patterns and return a list of waypoints:
    - 'minor': Small legitimate deviation (e.g., local detour)
    - 'major': Large deviation (potentially suspicious)
    - 'loop': Creates a loop in the route (suspicious)
    - 'double_loop': Creates two loops (highly suspicious)
    - 'random': Random choice of the above
    """
    if deviation_type == "random":
        deviation_type = random.choice(["minor", "major", "loop", "double_loop"])
    
    if deviation_type == "minor":
        # Minor deviation - just a small shift from midpoint
        mid_point = calculate_realistic_mid_deviation(origin, destination)
        return [mid_point]
    
    elif deviation_type == "major":
        # Major deviation - significant detour
        mid_point = calculate_realistic_mid_deviation(origin, destination)
        # complex deviation
        second_point = calculate_realistic_mid_deviation(mid_point, destination)
        return [mid_point, second_point]
    
    
    elif deviation_type == "loop":
        # Create a loop by adding waypoints that force a circuit to signfy additional distance covered along a circle 
        direct_vector = (destination[0] - origin[0], destination[1] - origin[1])
        distance = math.sqrt(direct_vector[0]**2 + direct_vector[1]**2)
        
        # Calculate three points to form a loop
        third_dist = distance * 0.33 
        two_third_dist = distance * 0.66  
        
        # Points along the direct path
        point1 = (origin[0] + direct_vector[0] * 0.33, origin[1] + direct_vector[1] * 0.33)
        point3 = (origin[0] + direct_vector[0] * 0.66, origin[1] + direct_vector[1] * 0.66)
        
        # Create a perpendicular point for loop
        perp_vector = (-direct_vector[1], direct_vector[0])  # Perpendicular vector
        perp_magnitude = math.sqrt(perp_vector[0]**2 + perp_vector[1]**2)
        loop_size = distance * random.uniform(0.2, 0.4)  # Size of the loop
        
        # Normalization and scaling of the vector
        if perp_magnitude > 0:
            norm_perp = (perp_vector[0] / perp_magnitude, perp_vector[1] / perp_magnitude)
            point2 = (point1[0] + norm_perp[0] * loop_size, point1[1] + norm_perp[1] * loop_size)
            
            return [point1, point2, point3]
        else:
            return [calculate_realistic_mid_deviation(origin, destination)]
    
    elif deviation_type == "double_loop":
        # Create two loops
        direct_vector = (destination[0] - origin[0], destination[1] - origin[1])
        distance = math.sqrt(direct_vector[0]**2 + direct_vector[1]**2)
        
        # First loop in the first half of the journey
        first_quarter = (origin[0] + direct_vector[0] * 0.25, origin[1] + direct_vector[1] * 0.25)
        third_quarter = (origin[0] + direct_vector[0] * 0.75, origin[1] + direct_vector[1] * 0.75)
        
        # Create perpendicular points for loops
        perp_vector = (-direct_vector[1], direct_vector[0])
        perp_magnitude = math.sqrt(perp_vector[0]**2 + perp_vector[1]**2)
        
        if perp_magnitude > 0:
            norm_perp = (perp_vector[0] / perp_magnitude, perp_vector[1] / perp_magnitude)
            
            # First loop
            loop1_size = distance * random.uniform(0.15, 0.25)
            loop1_point = (first_quarter[0] + norm_perp[0] * loop1_size, 
                           first_quarter[1] + norm_perp[1] * loop1_size)
            
            # Second loop (in opposite direction)
            loop2_size = distance * random.uniform(0.15, 0.25)
            loop2_point = (third_quarter[0] - norm_perp[0] * loop2_size,
                           third_quarter[1] - norm_perp[1] * loop2_size)
            
            return [first_quarter, loop1_point, third_quarter, loop2_point]
        else:
            # Fallback
            return [calculate_realistic_mid_deviation(origin, destination)]
    
    # Fallback for any other case
    return [calculate_realistic_mid_deviation(origin, destination)]

def get_osrm_route(start_coords: Tuple[float, float], end_coords: Tuple[float, float], 
                   waypoints: List[Tuple[float, float]] = None,
                   geometry_only=False,
                   retry_count=3) -> Dict[str, Any]:
    """
    Get routing information from OSRM service.
    Returns time (seconds), distance (meters), and route geometry.
    
    Parameters:
    - start_coords: (lat, lon) of starting point
    - end_coords: (lat, lon) of destination
    - waypoints: List of (lat, lon) points to include in the route
    
    Returns dictionary with distance_m, duration_s, and geometry"""
                       
    # Build coordinates string for OSRM ((lon,lat) format)
    coords = f"{start_coords[1]},{start_coords[0]}"
    
    if waypoints:
        for wp in waypoints:
            coords += f";{wp[1]},{wp[0]}"
            
    coords += f";{end_coords[1]},{end_coords[0]}"
    
    # Configure OSRM API URL
    base_url = "http://router.project-osrm.org/route/v1/driving/"
    full_url = f"{base_url}{coords}"
    params = {
        "overview": "full",  # Get full route geometry
        "geometries": "geojson",  # GeoJSON format for coordinates
        "annotations": "true"  # Get speed and duration for each segment
    }
    
    # Try to get route from OSRM API
    for attempt in range(retry_count):
        try:
            response = requests.get(full_url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data["code"] == "Ok" and len(data["routes"]) > 0:
                    route = data["routes"][0]
                    
                    # If we only need geometry, return just the coordinates
                    if geometry_only:
                        # Convert from [lon, lat] to [lat, lon] for folium
                        return [(coord[1], coord[0]) for coord in route["geometry"]["coordinates"]]
                    
                    return {
                        "distance_m": route["distance"],             # meters
                        "duration_s": route["duration"],             # seconds
                        "geometry": route["geometry"]["coordinates"] # list of [lon, lat] points
                    }
            else:
                print(f"OSRM API error: {response.status_code}")
        except Exception as e:
            print(f"OSRM request error (attempt {attempt+1}/{retry_count}): {e}")
        
        time.sleep(1)
    
    # If all retries failed, calculate rough estimates
    print("Warning: OSRM request failed. Using fallback estimation.")
    
    # Calculate rough distance using geodesic method
    distance_m = geodesic((start_coords[0], start_coords[1]), 
                           (end_coords[0], end_coords[1])).meters
    
    #simple estimated duration (trucks average 60 km/h)
    duration_s = distance_m / (60 * 1000 / 3600)  # Convert 60 km/h to m/s
    
    # If we only need geometry for visualization
    if geometry_only:
        return [(start_coords[0], start_coords[1]), (end_coords[0], end_coords[1])]
    
    return {
        "distance_m": distance_m,
        "duration_s": duration_s,
        "geometry": [[start_coords[1], start_coords[0]], [end_coords[1], end_coords[0]]]
    }

def calculate_route_similarity(actual_route, expected_route) -> float:
    """
    Calculate similarity between actual and expected routes.
    
    Parameters:
    - actual_route: List of coordinates [(lon, lat), ...] from actual route
    - expected_route: List of coordinates [(lon, lat), ...] from expected route
    
    Returns a similarity score between 0 and 1, where:
    - 1.0: Routes are nearly identical
    - >0.5: Less than half the pathway is deviated 
    - <0.5: More than half the pathway is deviated
    - Close to 0: The entire route is different
    """
    if not actual_route or not expected_route:
        # Fallback if route data is missing
        return 0.5 
    
    # Ensure we have meaningful routes to compare
    if len(actual_route) < 2 or len(expected_route) < 2:
        return 0.5 
    
    # Convert from [lon, lat] to [lat, lon]
    if isinstance(actual_route[0], list):
        actual_points = [(coord[1], coord[0]) for coord in actual_route]
    else:
        actual_points = actual_route
        
    if isinstance(expected_route[0], list):
        expected_points = [(coord[1], coord[0]) for coord in expected_route]
    else:
        expected_points = expected_route
    
    # For identical routes, return 1.0 immediately
    if actual_points == expected_points:
        return 1.0
        
    # First check if routes are significantly similar in terms of start, end, and length
    actual_start = actual_points[0]
    actual_end = actual_points[-1]
    expected_start = expected_points[0]
    expected_end = expected_points[-1]
    
    # Check if start and end are identical (common case for OSRM routes)
    start_identical = geodesic(actual_start, expected_start).km < 0.1  # Within 100m
    end_identical = geodesic(actual_end, expected_end).km < 0.1  # Within 100m
    
    if start_identical and end_identical:
        # If start and end match, check if general path is similar
        
        # Calculate total distance of each route
        actual_length = sum(geodesic(actual_points[i], actual_points[i+1]).km 
                           for i in range(len(actual_points)-1))
        expected_length = sum(geodesic(expected_points[i], expected_points[i+1]).km 
                             for i in range(len(expected_points)-1))
        
        # Calculate length ratio (always ≤ 1.0)
        length_ratio = min(actual_length, expected_length) / max(actual_length, expected_length)
        
        # If lengths are very similar (within 5%), and starts/ends match,
        if length_ratio > 0.95:
            # Sample some midpoints to verify
            actual_mid = actual_points[len(actual_points)//2]
            expected_mid = expected_points[len(expected_points)//2]
            
            mid_distance = geodesic(actual_mid, expected_mid).km
            
            # If midpoints are also close, routes are very similar
            if mid_distance < 5.0:  # Within 5km (for highways this is reasonable)
                return max(0.9, length_ratio)  # Very high similarity

    # For more complex comparisons, use Fréchet distance approximation
    # Sample both routes to the same number of points
    sample_size = 50 
    
    # Create evenly spaced samples
    actual_samples = []
    for i in range(sample_size):
        idx = min(int(i * len(actual_points) / sample_size), len(actual_points) - 1)
        actual_samples.append(actual_points[idx])
        
    expected_samples = []
    for i in range(sample_size):
        idx = min(int(i * len(expected_points) / sample_size), len(expected_points) - 1)
        expected_samples.append(expected_points[idx])
    
    # Calculate closest point distances (simpler Fréchet distance approximation)
    total_distance = 0
    max_distance = 0
    
    for i in range(len(actual_samples)):
        # Find closest point in expected route to each point in actual route
        min_dist = float('inf')
        for j in range(len(expected_samples)):
            dist = geodesic(actual_samples[i], expected_samples[j]).km
            min_dist = min(min_dist, dist)
        
        total_distance += min_dist
        max_distance = max(max_distance, min_dist)
    
    # Calculate average distance
    avg_distance = total_distance / len(actual_samples)
    
    """ Convert distance to similarity (inversely related)
     Use exponential decay to give high scores to routes with small average distances
     and low scores to routes with large distances 
    """
    
    """ Parameters tuned to produce desired scores
    - Nearly identical routes should have avg_distance near 0 (score -> 1.0)
    - Completely different routes might have avg_distance of 50km+ (score -> 0.0)
    - Partially similar routes should have proportional scores 
    """
    similarity = math.exp(-avg_distance / 10.0)
    
    # Adjust similarity slightly based on length ratio
    actual_length = sum(geodesic(actual_samples[i], actual_samples[i+1]).km 
                       for i in range(len(actual_samples)-1))
    expected_length = sum(geodesic(expected_samples[i], expected_samples[i+1]).km 
                         for i in range(len(expected_samples)-1))
    
    length_ratio = min(actual_length, expected_length) / max(actual_length, expected_length)
    
    # Final score: combine similarity and length ratio (90/10 weighted)
    final_score = 0.9 * similarity + 0.1 * length_ratio

    """
    Ensure scores align with requirements:
    If all route points match exactly: score should be 1.0
    If less than half deviates: score should be > 0.5
    If more than half deviates: score should be < 0.5
    If routes entirely different: score should be close to 0.0
    """
    if final_score > 0.95:
        # If very similar, round up to nearly 1.0
        return 0.98 + (final_score - 0.95) * 0.4
    elif final_score > 0.8:
        # Slightly boost high similarity scores
        return 0.8 + (final_score - 0.8) * 1.5
    elif final_score < 0.2:
        # Slightly decrease very low similarity scores
        return final_score * 0.8
    else:
        # Keep middle range as is
        return final_score

def calculate_loop_count(route_coords) -> int:
    """
    Detect loops in a route by analyzing self-intersections and direction changes.
    
    Parameters:
    - route_coords: List of coordinates representing the route
    
    Returns an integer count of detected loops.
    """
    if not route_coords or len(route_coords) < 10:
        # Not enough points to detect loops reliably
        return random.randint(0, 1)
    
    # Convert from [lon, lat] to [lat, lon] if needed
    if isinstance(route_coords[0], list):
        coords = [(point[1], point[0]) for point in route_coords]
    else:
        coords = route_coords
    
    # Simplify the route to key points to avoid noise
    # Take every Nth point based on route length
    step = max(1, len(coords) // 100)
    simplified_route = coords[::step]
    
    if len(simplified_route) < 5:
        simplified_route = coords  # Too few points, use original
    
    # Method 1: Direction changes analysis
    direction_changes = 0
    significant_turns = 0
    prev_direction = None
    
    for i in range(len(simplified_route) - 2):
        pt1 = simplified_route[i]
        pt2 = simplified_route[i+1]
        
        # Calculate direction vector
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        
        # Calculate bearing
        direction = math.atan2(dy, dx)
        
        if prev_direction is not None:
            # Check for significant direction change
            diff = abs(direction - prev_direction)
            
            # Normalize to -π to π
            if diff > math.pi:
                diff = 2 * math.pi - diff
                
            # Count significant turns (> 45 degrees)
            if diff > math.pi/4:
                direction_changes += 1
                
            # Count very sharp turns (> 90 degrees)
            if diff > math.pi/2:
                significant_turns += 1
        
        prev_direction = direction
    
    # Method 2: Self-intersection detection
    intersections = 0
    
    # Check for potential route self-intersections (simplified approach)
    for i in range(len(simplified_route) - 3):
        for j in range(i + 2, len(simplified_route) - 1):
            # Check if segments (i,i+1) and (j,j+1) could intersect
            segment1 = (simplified_route[i], simplified_route[i+1])
            segment2 = (simplified_route[j], simplified_route[j+1])
            
            # Calculate bounding boxes
            s1_min_lat = min(segment1[0][0], segment1[1][0])
            s1_max_lat = max(segment1[0][0], segment1[1][0])
            s1_min_lon = min(segment1[0][1], segment1[1][1])
            s1_max_lon = max(segment1[0][1], segment1[1][1])
            
            s2_min_lat = min(segment2[0][0], segment2[1][0])
            s2_max_lat = max(segment2[0][0], segment2[1][0])
            s2_min_lon = min(segment2[0][1], segment2[1][1])
            s2_max_lon = max(segment2[0][1], segment2[1][1])
            
            # Check if bounding boxes overlap
            if (s1_max_lat >= s2_min_lat and s1_min_lat <= s2_max_lat and
                s1_max_lon >= s2_min_lon and s1_min_lon <= s2_max_lon):
                # Potential intersection detected
                intersections += 1
    
    """Combine methods to estimate loop count
    Simplified heuristic:
    1. Significant direction changes contribute to loop detection
    2. Self-intersections are strong indicators of loops
    """
    # Weight the factors to determine loop count
    loop_indicators = (direction_changes // 4) + significant_turns + (intersections // 2)
    
    # Convert to final loop count (with some randomness for variation)
    loops = min(5, int(loop_indicators / 3) + random.randint(0, 1))
    
    return loops

def visualize_delivery(delivery_data, filename):
    """
    Create a folium map visualization of the delivery showing both optimal and actual routes.
    Also highlights detected loops in the route.
    
    Parameters:
    - delivery_data: Dictionary containing delivery information
    - filename: Output HTML filename for the map
    """
    # Extract route data
    start_coords = (delivery_data["origin_lat"], delivery_data["origin_lon"])
    end_coords = (delivery_data["destination_lat"], delivery_data["destination_lon"])
    optimal_route = delivery_data["optimal_route_coords"]
    actual_route = delivery_data["actual_route_coords"]
    
    # Calculate map center (use midpoint of the optimal route)
    if optimal_route:
        mid_index = len(optimal_route) // 2
        center = optimal_route[mid_index]
    else:
        # Fallback to average of start and end
        center = ((start_coords[0] + end_coords[0])/2, 
                 (start_coords[1] + end_coords[1])/2)
    
    # Create folium map
    m = folium.Map(location=center, zoom_start=6)
    
    # Add markers for start and end points
    folium.Marker(
        start_coords, 
        popup=f"Origin: {delivery_data['origin_city']}",
        icon=folium.Icon(color='green', icon='play', prefix='fa')
    ).add_to(m)
    
    folium.Marker(
        end_coords, 
        popup=f"Destination: {delivery_data['destination_city']}",
        icon=folium.Icon(color='red', icon='stop', prefix='fa')
    ).add_to(m)
    
    # Add the optimal route in blue
    if optimal_route:
        folium.PolyLine(
            optimal_route, 
            color='blue', 
            weight=3, 
            opacity=0.8,
            popup=f"Optimal Route: {delivery_data['distance_km']:.1f} km"
        ).add_to(m)
    
    if actual_route:
        folium.PolyLine(
            actual_route, 
            color='orange', 
            weight=3, 
            opacity=0.8,
            popup=f"Actual Route: {delivery_data['actual_distance_km']:.1f} km"
        ).add_to(m)
    

    has_loops = False
    if delivery_data["loop_count"] > 0 and len(actual_route) > 10:
        # Analyze the route to find potential loop segments
        loop_segments = identify_loop_segments(actual_route)
        
        # Add each loop segment as a highlighted overlay
        for i, segment in enumerate(loop_segments):
            if segment:
                has_loops = True
                folium.PolyLine(
                    segment,
                    color='purple',
                    weight=5,
                    opacity=0.9,
                    popup=f"Detected Loop #{i+1}"
                ).add_to(m)

    fulfillment_status = "Yes" if delivery_data.get("order_fulfilled", 0) == 1 else "No"
    
    html_info = f"""
    <div style='min-width:200px'>
        <h4 style='background-color:#f8f9fa; padding:8px; border-radius:4px;'>Delivery {delivery_data['delivery_id']}</h4>
        <div style='margin-bottom:10px;'>
            <b>Route:</b> {delivery_data['origin_city']} → {delivery_data['destination_city']}<br>
        </div>
        <div style='background-color:#f8f9fa; padding:8px; border-radius:4px; margin-bottom:10px;'>
            <b>Distance:</b><br>
            • Optimal: {delivery_data['distance_km']:.1f} km<br>
            • Actual: {delivery_data['actual_distance_km']:.1f} km<br>
            • Ratio: {delivery_data['actual_distance_km']/delivery_data['distance_km']:.2f}x
        </div>
        <div style='background-color:#f8f9fa; padding:8px; border-radius:4px; margin-bottom:10px;'>
            <b>Time:</b><br>
            • Expected: {delivery_data['expected_time_for_delivery']:.1f} min<br>
            • Actual: {delivery_data['actual_time_for_delivery']:.1f} min<br>
            • Ratio: {delivery_data['actual_time_for_delivery']/delivery_data['expected_time_for_delivery']:.2f}x
        </div>
        <div style='background-color:#f8f9fa; padding:8px; border-radius:4px; margin-bottom:10px;'>
            <b>Route Analysis:</b><br>
            • Loop Count: {delivery_data['loop_count']}<br>
            • Route Similarity: {delivery_data['route_similarity_score']:.2f}
        </div>
        <div style='background-color:{"#ffebee" if delivery_data["marked_as_suspicious"] == 1 else "#e8f5e9"}; padding:8px; border-radius:4px;'>
            <b>Delivery Status:</b><br>
            • Order Fulfilled: {fulfillment_status}<br>
            • <span style='color:{"red" if delivery_data["marked_as_suspicious"] == 1 else "green"};'><b>Suspicious:</b> {"Yes" if delivery_data['marked_as_suspicious'] == 1 else "No"}</span><br>
            • Fraud Type: {delivery_data.get('fraud_type', 'N/A')}
        </div>
    </div>
    """
    
    folium.Marker(
        center,
        popup=folium.Popup(html_info, max_width=300),
        icon=folium.Icon(color='purple', icon='info-sign')
    ).add_to(m)
    
 
    legend_html = '''
    <div style="position: fixed; 
        bottom: 50px; right: 50px; width: 180px; height: auto; 
        background-color: white; border:2px solid grey; z-index:9999; 
        font-size:14px; padding: 10px; border-radius: 6px;">
        <p style="margin-top: 0; font-weight: bold; text-align: center;">Legend</p>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="background-color: green; width: 15px; height: 15px; border-radius: 50%; margin-right: 10px;"></div>
            <span>Origin</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="background-color: red; width: 15px; height: 15px; border-radius: 50%; margin-right: 10px;"></div>
            <span>Destination</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="background-color: blue; width: 15px; height: 3px; margin-right: 10px;"></div>
            <span>Optimal Route (OSRM)</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="background-color: orange; width: 15px; height: 3px; margin-right: 10px;"></div>
            <span>Actual Route</span>
        </div>
    '''
    
    # Add loop legend item only if loops are present
    if has_loops:
        legend_html += '''
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="background-color: purple; width: 15px; height: 5px; margin-right: 10px;"></div>
            <span>Detected Loops</span>
        </div>
        '''
    
    legend_html += '''
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="background-color: purple; width: 15px; height: 15px; border-radius: 50%; margin-right: 10px;"></div>
            <span>Delivery Info</span>
        </div>
        <p style="margin-bottom: 0; font-size: 12px; color: #666;">Visualization via OSRM API</p>
    </div>
    '''
    
    m.get_root().html.add_child(folium.Element(legend_html))
    

    title_html = f'''
    <div style="position: fixed; 
        top: 10px; left: 50%; transform: translateX(-50%);
        background-color: white; border:2px solid grey;
        z-index:9999; font-size:16px; padding: 10px; border-radius: 6px;">
        <h3 style="margin: 0;">{delivery_data['origin_city']} to {delivery_data['destination_city']} Delivery</h3>
        <p style="margin: 5px 0 0; text-align: center; font-size: 12px;">
            Fraud Type: <span style="color: {'red' if delivery_data.get('fraud_type', 'legitimate') != 'legitimate' else 'green'};">
                {delivery_data.get('fraud_type', 'Legitimate')}
            </span>
        </p>
    </div>
    '''
    
    m.get_root().html.add_child(folium.Element(title_html))
    
    m.save(filename)
    
    return

def identify_loop_segments(route_coords):
    """
    Identify potential loop segments in a route for visualization.
    
    Parameters:
    - route_coords: List of (lat, lon) coordinates
    
    Returns list of loop segments (each segment is a list of coordinates)
    """
    if len(route_coords) < 10:
        return []
    
    # Simplify the route to avoid noise
    step = max(1, len(route_coords) // 100)
    simplified_route = route_coords[::step]
    
    if len(simplified_route) < 5:
        simplified_route = route_coords
    
    loop_segments = []
    
    # 1: Direction change detection
    prev_direction = None
    segment_start = 0
    current_segment = []
    significant_turns = 0
    
    for i in range(len(simplified_route) - 1):
        pt1 = simplified_route[i]
        pt2 = simplified_route[i+1]
        
        # Calculate direction vector
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        
        # Calculate bearing
        direction = math.atan2(dy, dx)
        
        if prev_direction is not None:
            # Check for significant direction change
            diff = abs(direction - prev_direction)
            
            # Normalize to -π to π
            if diff > math.pi:
                diff = 2 * math.pi - diff
                
            # Detect sharp turns (> 90 degrees)
            if diff > math.pi/2:
                significant_turns += 1
                current_segment.append(simplified_route[i])
                
                # If we've made multiple sharp turns in a segment, mark it as a potential loop
                if significant_turns >= 3 and len(current_segment) > 3:
                    # Close the loop
                    current_segment.append(simplified_route[i+1])
                    loop_segments.append(current_segment)
                    
                    # Reset for next loop detection
                    current_segment = []
                    significant_turns = 0
                    segment_start = i + 1
            else:
                if len(current_segment) > 0:
                    current_segment.append(simplified_route[i])
        else:
            # Start a new segment
            current_segment = [simplified_route[i]]
            
        prev_direction = direction
    
    # 2: Self-intersection detection
    for i in range(len(simplified_route) - 10):
        for j in range(i + 10, len(simplified_route) - 1):
            # Check if point j is very close to point i (potential loop)
            distance = geodesic(simplified_route[i], simplified_route[j]).km
            
            if distance < 1.0:  # Within 1km
                # Found a potential loop
                loop_segment = simplified_route[i:j+1]
                
                # Only add if it's a reasonably sized loop
                if len(loop_segment) >= 4 and distance > 0.1:
                    loop_segments.append(loop_segment)
    
    # Deduplicate and limit the number of loop segments to avoid excessive highlights
    unique_loops = []
    for segment in loop_segments:
        # Simple deduplication by checking start/end points
        if not any(segment[0] == s[0] and segment[-1] == s[-1] for s in unique_loops):
            unique_loops.append(segment)
    
    # Return at most 3 loops to avoid cluttering the map
    return unique_loops[:3]

def generate_synthetic_deliveries(num_rows=50, suspicious_percent=0.20, seed=42, create_visualizations=True):
    """
    Generate synthetic delivery data with coherent GPS coordinates and fraud patterns.
    
    Parameters:
    - num_rows: Number of deliveries to generate (default: 50)
    - suspicious_percent: Percentage of deliveries that should be suspicious (default: 20%)
    - seed: Random seed for reproducibility
    - create_visualizations: Whether to create HTML visualizations of routes
    
    Returns DataFrame with delivery data
    """
    np.random.seed(seed)
    random.seed(seed)
    
    if create_visualizations:
        os.makedirs("route_visualizations", exist_ok=True)
    
    # Create fraud driver IDs (consistent fraud patterns)
    num_fraud_drivers = int(num_rows * suspicious_percent * 0.7)  # 70% of fraud from consistent drivers
    fraud_drivers = []
    
    for i in range(num_fraud_drivers):
        fraud_pattern = random.choice([p["name"] for p in FRAUD_PATTERNS])
        fraud_drivers.append({
            "driver_id": f"D{1000 + i}",
            "fraud_pattern": fraud_pattern,
            "fraud_probability": random.uniform(0.7, 0.95)  # Not every delivery by fraud driver is fraudulent
        })
    
    data_rows = []
    
    # Generate routes with appropriate mix of legitimate and suspicious deliveries
    delivery_count = 0
    suspicious_count = 0
    suspicious_target = int(num_rows * suspicious_percent)
    
    print(f"Generating {num_rows} deliveries with ~{suspicious_percent:.0%} suspicious patterns...")
    
    while delivery_count < num_rows:
        # Get random origin-destination city pair
        origin_city, destination_city = get_city_pair()
        
        # Assign a driver (some will be fraud drivers)
        if suspicious_count < suspicious_target and random.random() < 0.85:  # Prioritize suspicious if below target
            # Choose a fraud driver
            driver = random.choice(fraud_drivers)
            is_fraud_delivery = random.random() < driver["fraud_probability"]
            current_driver_id = driver["driver_id"]
            fraud_type = driver["fraud_pattern"] if is_fraud_delivery else "legitimate"
        else:
            # Regular driver - much lower fraud probability
            current_driver_id = f"D{2000 + delivery_count}"
            is_fraud_delivery = random.random() < 0.03  # 3% chance of random fraud
            fraud_type = random.choice([p["name"] for p in FRAUD_PATTERNS]) if is_fraud_delivery else "legitimate"
        
        # Get coordinates
        origin = (origin_city["lat"], origin_city["lon"])
        destination = (destination_city["lat"], destination_city["lon"])
        
        # Get optimal route first (this will be our baseline)
        optimal_route_result = get_osrm_route(origin, destination)
        
        # Store coordinate arrays for visualization
        optimal_route_coords = [(coord[1], coord[0]) for coord in optimal_route_result["geometry"]]
        
        # Calculate baseline metrics
        distance_km = optimal_route_result["distance_m"] / 1000  # in km
        expected_time_minutes = optimal_route_result["duration_s"] / 60  # in minutes
        
        # Determine route deviation pattern based on fraud type
        waypoints = []
        reached_destination = True  # Default for legitimate deliveries
        
        if is_fraud_delivery:
            if fraud_type == "route_padder":
                # More complex deviations for route padders
                deviation_type = random.choice(["loop", "double_loop"])
                waypoints = create_route_deviation(origin, destination, deviation_type)
                reached_destination = True  # They still reach the destination
                
            elif fraud_type == "time_waster":
                # Small or moderate deviations for time wasters
                deviation_type = random.choice(["minor", "major"])
                waypoints = create_route_deviation(origin, destination, deviation_type)
                reached_destination = True  # They still reach the destination
                
            elif fraud_type == "ghost_delivery":
                # Ghost deliveries sometimes have unrealistic routes
                if random.random() < 0.6:
                    # Most ghost deliveries don't reach destination
                    # Create a route that stops short of the destination
                    mid_point = (
                        origin[0] + (destination[0] - origin[0]) * random.uniform(0.5, 0.85),
                        origin[1] + (destination[1] - origin[1]) * random.uniform(0.5, 0.85)
                    )
                    waypoints = [mid_point]
                    reached_destination = False
                else:
                    # Some claim bizarre routes
                    waypoints = create_route_deviation(origin, destination, "major")
                    # Randomly determine if they actually reached destination
                    reached_destination = random.random() < 0.3
        else:
            # Legitimate delivery might have small deviations
            if random.random() < 0.3:  # 30% chance of small detour
                waypoints = create_route_deviation(origin, destination, "minor")
                reached_destination = True
        
        # Generate weather and traffic conditions
        weather_sev = random.uniform(0.0, 1.0)
        traffic_cong = random.uniform(0.0, 1.0)
        
        # Get actual route (with waypoints if any)
        if waypoints:
            actual_route_result = get_osrm_route(origin, destination, waypoints)
        else:
            # For legitimate or "ghost" deliveries with no waypoints
            actual_route_result = optimal_route_result.copy()
            
            # For ghost deliveries with direct route, adjust times
            if is_fraud_delivery and fraud_type == "ghost_delivery" and not waypoints:
                # Unrealistically fast time
                actual_route_result["duration_s"] *= random.uniform(0.3, 0.5)
        
        # Store actual route coordinates for visualization
        actual_route_coords = [(coord[1], coord[0]) for coord in actual_route_result["geometry"]]
        
        # For ghost deliveries, potentially modify the final destination point
        if is_fraud_delivery and fraud_type == "ghost_delivery" and not reached_destination:
            # Replace the final coordinates with a point that's not at the destination
            if actual_route_coords and len(actual_route_coords) > 1:
                # Use the second-to-last point in the route
                actual_route_coords[-1] = actual_route_coords[-2]
        
        # Calculate actual metrics
        actual_distance_km = actual_route_result["distance_m"] / 1000
        
        # Calculate time factors depending on delivery type
        if is_fraud_delivery:
            if fraud_type == "route_padder":
                # Route padders have times proportional to their longer distance
                time_factor = random.uniform(0.9, 1.1) * (actual_distance_km / distance_km)
            elif fraud_type == "time_waster":
                # Time wasters report much longer times than needed
                time_factor = random.uniform(1.7, 3.0)
            elif fraud_type == "ghost_delivery":
                # Ghost deliveries are unrealistically fast
                time_factor = random.uniform(0.2, 0.5)
        else:
            # Regular deliveries have reasonable time variations
            time_factor = random.uniform(0.8, 1.2)
            
            # Adjust for weather and traffic if severe
            if weather_sev > 0.7:  # Bad weather
                time_factor *= random.uniform(1.1, 1.4)
            
            if traffic_cong > 0.7:  # Heavy traffic
                time_factor *= random.uniform(1.1, 1.3)
        
        # Calculate actual delivery time
        actual_time_minutes = expected_time_minutes * time_factor
        
        # For route padders and legitimate deliveries, time should make sense with distance
        if fraud_type == "route_padder" or fraud_type == "legitimate":
            # Ensure time is at least somewhat proportional to distance
            min_time = actual_distance_km * 0.6  # Minimum 36 km/h
            actual_time_minutes = max(actual_time_minutes, min_time)
        
        # Calculate route similarity based on actual and optimal route geometries
        route_similarity = calculate_route_similarity(
            actual_route_result["geometry"], 
            optimal_route_result["geometry"]
        )
        
        # Calculate loop count based on actual route
        loop_count = calculate_loop_count(actual_route_result["geometry"])
        
        # Immediately flag extreme route deviations (similarity < 0.2) as route fraud
        if route_similarity < 0.2 and not is_fraud_delivery:
            # Extreme route deviation - flag as route padding fraud immediately
            fraud_type = "route_padder"
            is_fraud_delivery = True

        # Flag different routes with similar/less delivery times as route fraud
        # (For routes with similarity between 0.2 and 0.4)
        if not is_fraud_delivery and route_similarity >= 0.2 and route_similarity < 0.4:
            if actual_distance_km > distance_km * 1.3:
                # Route is very different AND much longer distance
                if actual_time_minutes <= expected_time_minutes * 1.1:
                    # But reported time is suspiciously similar or less than expected
                    fraud_type = "route_padder"
                    is_fraud_delivery = True

        # NEW RULE 3: Define early delivery threshold with both percentage and absolute time criteria
        is_normal_early = actual_time_minutes < expected_time_minutes * 0.85  # 15% earlier than expected

        # Check for unrealistically early arrivals using both percentage and absolute time difference
        is_unrealistic_early = False

        # Percentage-based check (more than 60% faster than expected)
        if actual_time_minutes < expected_time_minutes * 0.4:
            is_unrealistic_early = True
        
        # Absolute time difference check (3+ hours early for medium/longer trips)
        time_difference_minutes = expected_time_minutes - actual_time_minutes
        if time_difference_minutes >= 180:  # 3 hours or more early (180 minutes)
            # Only apply this criterion for deliveries expected to take more than 4 hours
            if expected_time_minutes >= 240:  # 4 hours
                is_unrealistic_early = True

        # For early deliveries that might otherwise be caught by existing rules,
        # only mark as fraud if they're unrealistically early
        if (is_normal_early or is_unrealistic_early) and not is_fraud_delivery:
            # First assume not fraud
            compelling_reason_for_fraud = False
            
            # Check for compelling reasons - unrealistically early is now enough
            if is_unrealistic_early:
                compelling_reason_for_fraud = True
            
            # Only mark as fraud if there's a compelling reason
            if compelling_reason_for_fraud:
                fraud_type = "ghost_delivery"
                is_fraud_delivery = True
            else:
                # Normal early delivery but not unrealistically early
                # Explicitly ensure it's not flagged as fraud
                is_fraud_delivery = False
                fraud_type = "legitimate"

        # After these new rules, ALL existing fraud detection logic continues as-is
        # with no modifications whatsoever
        
        # Use ghost delivery logic BEFORE the order fulfillment determination
        if is_fraud_delivery and fraud_type == "ghost_delivery":
            # For pre-classified ghost deliveries, use the reached_destination flag
            # that was already set in the route generation
            pass
        elif not is_fraud_delivery and random.random() < 0.03:
            # Small chance that even legitimate deliveries might not reach destination
            reached_destination = False
        
        # Always mark as ghost delivery if destination not reached, regardless of fraud type
        if not reached_destination and fraud_type != "ghost_delivery":
            fraud_type = "ghost_delivery"
            is_fraud_delivery = True
        
        # Determine driver rating based on fraud type
        if is_fraud_delivery:
            pattern = next((p for p in FRAUD_PATTERNS if p["name"] == fraud_type), None)
            if pattern:
                driver_rating = random.uniform(pattern.get("driver_rating", (1.0, 3.0))[0], 
                                             pattern.get("driver_rating", (1.0, 3.0))[1])
            else:
                # Fallback if pattern not found
                driver_rating = random.uniform(1.0, 3.0)
        else:
            # Legitimate deliveries have better ratings
            driver_rating = random.uniform(3.0, 5.0)
        
        # Determine order fulfillment based on fraud type and destination status
        if fraud_type == "ghost_delivery":
            # For ghost deliveries:
            # 1. If destination not reached: order is never fulfilled
            # 2. If destination reached: still very low chance of fulfillment
            if not reached_destination:
                order_fulfilled = 0
            else:
                # Even if reached destination, ghost delivery typically means order not fulfilled but allow a very small chance
                order_fulfilled = 1 if random.random() < 0.1 else 0
        elif is_fraud_delivery:
            # For other fraud types
            pattern = next((p for p in FRAUD_PATTERNS if p["name"] == fraud_type), None)
            fulfillment_prob = pattern.get("fulfillment_probability", 0.5) if pattern else 0.5
            order_fulfilled = 1 if random.random() < fulfillment_prob else 0
        else:
            # Legitimate deliveries
            order_fulfilled = 1 if random.random() < 0.95 else 0
        
        # If order is not fulfilled, classify as ghost delivery regardless of whether destination was reached
        if order_fulfilled == 0 and fraud_type != "ghost_delivery":
            fraud_type = "ghost_delivery"
            is_fraud_delivery = True
            
            # Make suspicious if it nbot marked earlier
            if not is_fraud_delivery:
                suspicious_count += 1
        
        actual_time_minutes = max(5, min(1440, actual_time_minutes))  # 5 min to 24 hours
        driver_rating = max(1.0, min(5.0, driver_rating))
        
        delivery_row = {
            "delivery_id": f"DEL-{delivery_count+1:03d}",
            "driver_id": current_driver_id,
            "origin_city": origin_city["name"],
            "destination_city": destination_city["name"],
            "expected_time_for_delivery": expected_time_minutes,
            "actual_time_for_delivery": actual_time_minutes,
            "distance_km": distance_km,
            "actual_distance_km": actual_distance_km,
            "loop_count": loop_count,
            "route_similarity_score": route_similarity,
            "weather_condition_severity": weather_sev,
            "traffic_congestion_level": traffic_cong,
            "driver_rating_and_credibility": driver_rating,
            "order_fulfilled": order_fulfilled,  # Binary fulfillment status
            "destination_reached": 1 if reached_destination else 0,  # Flag if destination was reached
            "origin_lat": origin[0],
            "origin_lon": origin[1],
            "destination_lat": destination[0],
            "destination_lon": destination[1],
            "marked_as_suspicious": 1 if is_fraud_delivery else 0,
            "fraud_type": fraud_type,
            # Store route coordinates for visualization
            "optimal_route_coords": optimal_route_coords,
            "actual_route_coords": actual_route_coords
        }
        
        # Add derived features
        delivery_row["delay_ratio"] = delivery_row["actual_time_for_delivery"] / delivery_row["expected_time_for_delivery"]
        delivery_row["distance_ratio"] = delivery_row["actual_distance_km"] / delivery_row["distance_km"]
        
        # Add to collection
        data_rows.append(delivery_row)
        
        # Update counters
        delivery_count += 1
        if is_fraud_delivery:
            suspicious_count += 1
        
        # Create visualization for this delivery
        if create_visualizations:
            viz_filename = f"route_visualizations/delivery_{delivery_row['delivery_id']}.html"
            visualize_delivery(delivery_row, viz_filename)
            print(f"Generated visualization {delivery_count}/{num_rows}: {viz_filename}")
    
    # Create DataFrame from all delivery data
    df = pd.DataFrame(data_rows)
    
    df_for_csv = df.drop(columns=["optimal_route_coords", "actual_route_coords"])
    
    return df_for_csv

if __name__ == "__main__":
   
    num_deliveries = 50  # Generate 50 deliveries
    suspicious_percentage = 0.20  # Target ~20% suspicious
    random_seed = 42  # For reproducibility
    
    print("Generating synthetic delivery data with OSRM-based routing...")
    
    synthetic_data = generate_synthetic_deliveries(
        num_rows=num_deliveries, 
        suspicious_percent=suspicious_percentage,
        seed=random_seed,
        create_visualizations=True
    )
    
    synthetic_data.to_csv("synthetic_delivery_data.csv", index=False)
    print("\nData generation complete!")
    
    print(f"Total deliveries: {len(synthetic_data)}")
    print(f"Suspicious deliveries: {synthetic_data['marked_as_suspicious'].sum()} ({synthetic_data['marked_as_suspicious'].mean():.1%})")
    
    if synthetic_data['marked_as_suspicious'].sum() > 0:
        fraud_breakdown = synthetic_data[synthetic_data['marked_as_suspicious']==1]['fraud_type'].value_counts()
        print("\nFraud types breakdown:")
        for fraud_type, count in fraud_breakdown.items():
            print(f"  - {fraud_type}: {count} deliveries ({count/synthetic_data['marked_as_suspicious'].sum():.1%})")
    
    print("\nVisualization files saved in the 'route_visualizations' directory")
    print("Synthetic data saved to 'synthetic_delivery_data.csv'")
