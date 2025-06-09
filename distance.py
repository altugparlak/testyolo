from math import radians, sin, cos, sqrt, atan2

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points
    on the Earth specified in decimal degrees.
    
    Returns distance in meters.
    """
    # Radius of Earth in meters
    R = 6371000  

    # Convert coordinates from degrees to radians
    φ1 = radians(lat1)
    φ2 = radians(lat2)
    Δφ = radians(lat2 - lat1)
    Δλ = radians(lon2 - lon1)

    # Haversine formula
    a = sin(Δφ / 2) ** 2 + cos(φ1) * cos(φ2) * sin(Δλ / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance


if __name__ == "__main__":
    lat1, lon1 = 41.19579890700771, 29.243817714760482
    lat2, lon2 = 41.19581483630624, 29.243894606112903
    lat3, lon3 = 41.1957439508858, 29.243867070434806

    dist = haversine_distance(lat1, lon1, lat2, lon2)
    dist2 = haversine_distance(lat1, lon1, lat3, lon3)
    print(f"Distance between point 1 and point 2: {dist:.2f} meters")
    print(f"Distance between point 1 and point 3: {dist2:.2f} meters")
    dist = haversine_distance(lat2, lon2, lat3, lon3)
    print(f"Distance between point 2 and point 3: {dist:.2f} meters")
